import torch
from torch import nn
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.draw import line_nd

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class InteractiveNnUNetTrainer(nnUNetTrainer):
    """
    Vollständiger interaktiver nnU-Net Trainer für industrielle CT-Segmentierung.
    Optimiert für nnU-Net V2.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: str = None, device: torch.device = torch.device('cpu')):
        # Parent init
        super().__init__(plans, configuration, fold, dataset_json=dataset_json, device=device)
        
        # Interaktions-Parameter
        self.segmentation_mode = 'instance'
        self.point_radius = 4
        self.scribble_thickness = 3
        self.num_epochs = 1
        self.num_processes_splitting = 1
        
        # Curriculum Learning Parameter
        self.min_interaction_rounds = 1  # WICHTIG: War im Originalcode nicht definiert
        self.max_interaction_rounds = 5
        
        # Wahrscheinlichkeiten für Simulation
        self.prob_bbox = 0.7
        self.prob_points = 0.95
        self.prob_scribbles = 0.4
        
        self.interaction_stats = {'bbox': 0, 'points_pos': 0, 'points_neg': 0, 'scribbles': 0}

    def initialize(self):
            if not self.was_initialized:
                # 1. Konfiguration laden
                configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
                self.batch_size = configuration_manager.batch_size
                self.patch_size = configuration_manager.patch_size
                
                # 2. Force CPU
                self.device = torch.device('cpu')
                
                # 3. Label Manager Setup
                self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
                
                # 4. Kanäle berechnen - Basierend auf deiner JSON
                # Wir nehmen die Anzahl der Einträge in "modality"
                if 'modality' in self.dataset_json:
                    self.original_input_channels = len(self.dataset_json['modality'])
                else:
                    self.original_input_channels = 1 # Fallback für CT
                
                # WICHTIG: Anzahl der Vordergrund-Klassen (Labels ohne Background)
                # In deiner JSON sind es 4: aluminum, plastic, steel, air
                num_fg_classes = len(self.dataset_json['labels']) - 1
                
                # Interaktions-Kanäle: 7 pro Klasse (Punkte, Distanz-Maps etc.)
                self.num_interaction_channels = 7 * num_fg_classes
                self.num_input_channels = self.original_input_channels + self.num_interaction_channels
                
                # 5. Architektur bauen
                self.build_network_architecture()
                
                # 6. Restliche Initialisierung
                self._build_loss()
                self.optimizer, self.lr_scheduler = self.configure_optimizers()
                self.grad_scaler = None # Zwingend für CPU
                
                self.was_initialized = True
                
                print("--- INITIALISIERUNG ERFOLGREICH ---")
                print(f"Bilder-Kanäle: {self.original_input_channels}")
                print(f"Vordergrund-Klassen: {num_fg_classes}")
                print(f"Interaktions-Kanäle: {self.num_interaction_channels}")
                print(f"Netzwerk-Input: {self.num_input_channels} Kanäle")
                print(f"Training läuft auf: {self.device}")
                print("-----------------------------------")

    def print_init_info(self):
        print(f"\n{'='*60}")
        print(f"Interactive nnU-Net Trainer Initialized (OPTIMIZED)")
        print(f"{'='*60}")
        print(f"  Mode:                           {self.segmentation_mode.upper()}")
        print(f"  Original image channels:        {self.original_input_channels}")
        print(f"  Interaction channels:           {self.num_interaction_channels}")
        print(f"  Total input channels:           {self.num_input_channels}")
        print(f"  Number of classes:              {self.label_manager.num_segmentation_heads}")
        print(f"  Network Class:                  {self.network.__class__.__name__}")
        print(f"{'='*60}\n")

    def build_network_architecture(self):
            """
            Baue Netzwerk dynamisch. 
            FIX: Konvertiert String-Operationen in ausführbare Torch-Klassen.
            """
            import torch.nn as nn
            from nnunetv2.utilities.network_initialization import InitWeights_He
            from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
            from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
            
            configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
            network_kwargs = configuration_manager.network_arch_init_kwargs
            
            # --- FIX FÜR 'STR' OBJECT IS NOT CALLABLE ---
            # Wir müssen sicherstellen, dass conv_op und norm_op echte Klassen sind, keine Strings
            if isinstance(network_kwargs.get('conv_op'), str):
                if 'Conv3d' in network_kwargs['conv_op']: network_kwargs['conv_op'] = nn.Conv3d
                elif 'Conv2d' in network_kwargs['conv_op']: network_kwargs['conv_op'] = nn.Conv2d
                
            if isinstance(network_kwargs.get('norm_op'), str):
                if 'InstanceNorm3d' in network_kwargs['norm_op']: network_kwargs['norm_op'] = nn.InstanceNorm3d
                elif 'InstanceNorm2d' in network_kwargs['norm_op']: network_kwargs['norm_op'] = nn.InstanceNorm2d
            
            # Wichtig: Falls 'nonlin' ein String ist
            if isinstance(network_kwargs.get('nonlin'), str):
                network_kwargs['nonlin'] = nn.LeakyReLU
            # --------------------------------------------

            network_kwargs['input_channels'] = self.num_input_channels
            network_kwargs['num_classes'] = self.label_manager.num_segmentation_heads
            
            arch_class_name = configuration_manager.network_arch_class_name
            mapping = {
                'PlainConvUNet': PlainConvUNet,
                'ResidualEncoderUNet': ResidualEncoderUNet
            }
            network_class = mapping.get(arch_class_name.split(".")[-1], PlainConvUNet)

            self.network = network_class(**network_kwargs)
            
            self.network.apply(InitWeights_He(1e-2))
            init_last_bn_before_add_to_0(self.network)
            
            if hasattr(self.network, 'seg_layers'):
                for layer in self.network.seg_layers:
                    layer.apply(init_last_bn_before_add_to_0)

            self.network.to(self.device)

    def _build_loss(self):
        self.loss = DC_and_CE_loss(
            soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={},
            weight_ce=1.0, weight_dice=1.0, ignore_label=None
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / self.num_epochs) ** 0.9)
        return optimizer, lr_scheduler

    # ==================== INTERAKTIONS-LOGIK (Unverändert gut) ====================
    
    def sample_points_from_mask(self, mask: np.ndarray, num_points: int, avoid_border: bool = True) -> List[Tuple[int, ...]]:
        if avoid_border:
            eroded = binary_dilation(~mask.astype(bool), iterations=3)
            valid_mask = mask & ~eroded
        else:
            valid_mask = mask
            
        coords = np.argwhere(valid_mask)
        if len(coords) == 0: coords = np.argwhere(mask)
        
        if len(coords) == 0: return []
        
        num_points = min(num_points, len(coords))
        indices = np.random.choice(len(coords), size=num_points, replace=False)
        return [tuple(coords[i]) for i in indices]

    def sample_error_based_points(self, gt_mask: np.ndarray, pred_mask: np.ndarray, num_points: int, positive: bool = True) -> List[Tuple[int, ...]]:
        error_mask = (gt_mask == 1) & (pred_mask == 0) if positive else (gt_mask == 0) & (pred_mask == 1)
        if error_mask.sum() == 0:
            return self.sample_points_from_mask(gt_mask if positive else ~gt_mask.astype(bool), num_points)
        return self.sample_points_from_mask(error_mask, num_points, avoid_border=False)

    def encode_point(self, point: Tuple[int, ...], shape: Tuple[int, ...], radius: int) -> np.ndarray:
        # Optimierung: Nur Bounding Box um den Punkt berechnen statt ganzes Array
        # (Hier vereinfacht als Full-Array belassen für Lesbarkeit, kann später optimiert werden)
        point_map = np.zeros(shape, dtype=np.float32)
        coords = np.ogrid[tuple(slice(0, s) for s in shape)]
        dist_sq = sum((c - p) ** 2 for c, p in zip(coords, point))
        point_map = np.exp(-dist_sq / (2 * (radius / 2.0) ** 2))
        return point_map

    def get_bounding_box(self, mask: np.ndarray, margin: int = 5) -> Optional[List[Tuple[int, int]]]:
        coords = np.argwhere(mask)
        if len(coords) == 0: return None
        return [(max(0, coords[:, d].min() - margin), min(mask.shape[d], coords[:, d].max() + margin + 1)) for d in range(mask.ndim)]

    def encode_bbox(self, bbox: List[Tuple[int, int]], shape: Tuple[int, ...]) -> np.ndarray:
        bbox_map = np.zeros(shape, dtype=np.float32)
        bbox_map[tuple(slice(s, e) for s, e in bbox)] = 1.0
        return bbox_map

    def create_scribble(self, mask: np.ndarray, num_strokes: int = 2) -> np.ndarray:
        scribble_map = np.zeros_like(mask, dtype=np.float32)
        coords = np.argwhere(mask)
        if len(coords) < 2: return scribble_map
        
        for _ in range(num_strokes):
            idx = np.random.choice(len(coords), size=min(2, len(coords)), replace=False)
            if len(idx) < 2: continue
            try:
                line_coords = line_nd(coords[idx[0]], coords[idx[1]], endpoint=True)
                scribble_map[line_coords] = 1.0
                if self.scribble_thickness > 1:
                    scribble_map = binary_dilation(scribble_map, iterations=self.scribble_thickness // 2).astype(np.float32)
            except: continue
        return scribble_map

    def simulate_interaction_round(self, gt_seg: np.ndarray, prev_pred: Optional[np.ndarray], round_num: int) -> np.ndarray:
        num_classes = self.label_manager.num_segmentation_heads
        interaction_maps = np.zeros((7 * num_classes, *gt_seg.shape), dtype=np.float32)
        
        for cls_idx in range(num_classes):
            cls_label = cls_idx + 1
            gt_mask = (gt_seg == cls_label).astype(np.uint8)
            if gt_mask.sum() == 0: continue
            
            pred_mask = (prev_pred == cls_label).astype(np.uint8) if prev_pred is not None else np.zeros_like(gt_mask)
            off = cls_idx * 7
            
            if round_num == 0:
                if random.random() < self.prob_bbox:
                    bbox = self.get_bounding_box(gt_mask, margin=random.randint(5, 10))
                    if bbox: 
                        interaction_maps[off + 1] = self.encode_bbox(bbox, gt_seg.shape)
                        self.interaction_stats['bbox'] += 1
                
                points = self.sample_points_from_mask(gt_mask, random.randint(2, 4))
                for p in points:
                    interaction_maps[off + 3] = np.maximum(interaction_maps[off + 3], self.encode_point(p, gt_seg.shape, self.point_radius))
                    self.interaction_stats['points_pos'] += 1
            else:
                # Refinement Rounds
                if random.random() < self.prob_points:
                    for p in self.sample_error_based_points(gt_mask, pred_mask, random.randint(2, 4), True):
                        interaction_maps[off + 3] = np.maximum(interaction_maps[off + 3], self.encode_point(p, gt_seg.shape, self.point_radius))
                        self.interaction_stats['points_pos'] += 1
                
                if random.random() < self.prob_points * 0.7:
                    for p in self.sample_error_based_points(gt_mask, pred_mask, random.randint(1, 3), False):
                        interaction_maps[off + 4] = np.maximum(interaction_maps[off + 4], self.encode_point(p, gt_seg.shape, self.point_radius))
                        self.interaction_stats['points_neg'] += 1
                        
                if random.random() < self.prob_scribbles:
                    interaction_maps[off + 5] = np.maximum(interaction_maps[off + 5], self.create_scribble(gt_mask))
                    self.interaction_stats['scribbles'] += 1
                    
        return interaction_maps

    # ==================== TRAIN STEP (OPTIMIZED) ====================

    def train_step(self, batch: dict) -> dict:
        # 1. Daten auf CPU schieben
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        # FIX: Target ist eine Liste. Wir schieben jedes Element einzeln auf das Device.
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Curriculum
        max_rounds = min(self.max_interaction_rounds, 1 + self.current_epoch // 30)
        num_rounds = random.randint(self.min_interaction_rounds, max_rounds)
        
        total_loss = 0
        current_pred_logits = None 
        
        for round_num in range(num_rounds):
            # Simulation vorbereiten
            prev_pred_discrete = None
            if current_pred_logits is not None:
                with torch.no_grad():
                    # Bei Deep Supervision nehmen wir immer die höchste Auflösung (Index 0)
                    if isinstance(current_pred_logits, (list, tuple)):
                        p_logits = current_pred_logits[0]
                    else:
                        p_logits = current_pred_logits
                    prev_pred_discrete = torch.argmax(p_logits, dim=1).cpu().numpy()
            
            # Interaktionssimulation
            batch_interactions = []
            for b in range(data.shape[0]):
                # WICHTIG: Ground Truth für Simulation ist immer target[0] (höchste Auflösung)
                gt_seg = target[0][b, 0].cpu().numpy() if isinstance(target, list) else target[b, 0].cpu().numpy()
                p_pred = prev_pred_discrete[b] if prev_pred_discrete is not None else None
                
                maps = self.simulate_interaction_round(gt_seg, p_pred, round_num)
                batch_interactions.append(maps)
            
            interaction_tensor = torch.from_numpy(np.stack(batch_interactions, axis=0)).to(self.device, dtype=torch.float32)
            network_input = torch.cat([data, interaction_tensor], dim=1)
            
            # Forward & Loss
            # enabled=False für Autocast, da wir auf CPU sind (spart Rechenlast/Fehler)
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                current_pred_logits = self.network(network_input)
                
                # nnU-Net Loss Funktion erwartet bei Deep Supervision Listen für Preds und Targets
                loss = self.loss(current_pred_logits, target)
            
            # Backward
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.detach()
        
        # Optimizer Step
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            
        return {'loss': total_loss / num_rounds}

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.current_epoch += 1
        self.interaction_stats = {k: 0 for k in self.interaction_stats}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        
        # Logging alle 10 Epochen
        if self.current_epoch % 10 == 0:
            avg_loss = np.mean([o['loss'].item() if torch.is_tensor(o['loss']) else o['loss'] for o in train_outputs])
            print(f" Epoch {self.current_epoch} | Loss: {avg_loss:.4f} | Interactions: {self.interaction_stats}")

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target'].to(self.device, non_blocking=True)
        
        # Validation immer nur Runde 0 (Initiale Interaktion)
        batch_interactions = []
        for b in range(data.shape[0]):
            gt_seg = target[b, 0].cpu().numpy()
            maps = self.simulate_interaction_round(gt_seg, None, round_num=0)
            batch_interactions.append(maps)
        
        interaction_tensor = torch.from_numpy(np.stack(batch_interactions)).to(self.device, dtype=torch.float32)
        network_input = torch.cat([data, interaction_tensor], dim=1)
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                pred = self.network(network_input)
                loss = self.loss(pred, target[:, 0].long())
                
        return {'loss': loss.detach()}

    # ==================== INFERENCE UTILS ====================

    def predict_with_interactions(self, image: np.ndarray, interactions: Dict[str, Dict[str, List]]) -> np.ndarray:
        self.network.eval()
        num_classes = self.label_manager.num_segmentation_heads
        shape = image.shape[1:]
        interaction_maps = np.zeros((7 * num_classes, *shape), dtype=np.float32)
        
        for cls_idx in range(num_classes):
            cls_key = f'class_{cls_idx + 1}'
            off = cls_idx * 7
            if cls_key not in interactions: continue
            
            data = interactions[cls_key]
            # Punkte kodieren
            for p in data.get('points_pos', []):
                interaction_maps[off + 3] = np.maximum(interaction_maps[off + 3], self.encode_point(p, shape, self.point_radius))
            for p in data.get('points_neg', []):
                interaction_maps[off + 4] = np.maximum(interaction_maps[off + 4], self.encode_point(p, shape, self.point_radius))
            # BBox
            if 'bbox' in data:
                interaction_maps[off + 1] = self.encode_bbox(data['bbox'], shape)
            # Scribbles
            if 'scribbles' in data:
                interaction_maps[off + 5] = data['scribbles']

        img_t = torch.from_numpy(image[None]).to(self.device, dtype=torch.float32)
        int_t = torch.from_numpy(interaction_maps[None]).to(self.device, dtype=torch.float32)
        inp = torch.cat([img_t, int_t], dim=1)
        
        with torch.no_grad():
            pred = self.network(inp)
            return torch.argmax(pred, dim=1)[0].cpu().numpy()

    # Save/Load Checkpoint methods are handled by parent class usually, 
    # but strictly needed if you want to save interaction params:
    def save_checkpoint(self, filename: str):
        state_dict = self.network.state_dict()
        optimizer_state = self.optimizer.state_dict()
        lr_scheduler_state = self.lr_scheduler.state_dict() if self.lr_scheduler else None
        
        torch.save({
            'network_state_dict': state_dict,
            'optimizer_state_dict': optimizer_state,
            'lr_scheduler_state_dict': lr_scheduler_state,
            'epoch': self.current_epoch,
            'interaction_params': {
                'point_radius': self.point_radius,
                'min_rounds': self.min_interaction_rounds,
                'max_rounds': self.max_interaction_rounds
            }
        }, filename)

    def load_checkpoint(self, filename: str):
        if not filename: return
        checkpoint = torch.load(filename, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['lr_scheduler_state_dict'] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        if 'interaction_params' in checkpoint:
            self.min_interaction_rounds = checkpoint['interaction_params'].get('min_rounds', 1)