import torch
from torch import nn
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import binary_dilation
from skimage.draw import line_nd

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
import torch
from torch import nn
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import binary_dilation
from skimage.draw import line_nd

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset



class InteractiveInstanceSegmentationTrainer(nnUNetTrainer):
    """
    Click-basierte Instance Segmentation für industrielle CT-Scans.
    
    FIXED: instance_mapping_path wird nach super().__init__ gesetzt
    """

    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict = None, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json=dataset_json, device=device)
        
        self.use_b2nd = True
        # Material-Klassen
        self.material_to_label = {
            'background': 0,
            'aluminum': 1,
            'plastic': 2,
            'steel': 3,
            'air': 4
        }
        self.label_to_material = {v: k for k, v in self.material_to_label.items()}
        self.num_materials = len(self.material_to_label) - 1  # Ohne Background
        
        # Instance mapping - wird später gesetzt
        self.instance_mapping_path = None
        self.instance_mappings = {}
        
        # Interaktions-Parameter
        self.point_radius = 4
        self.scribble_thickness = 3
        
        # Input: 1 (CT) + 4 (Materials) + 7 (Interactions) = 12
        self.num_input_channels = 1 + self.num_materials + 7
        
        # Output: 2 Klassen (Background, Object)
        self.num_output_classes = 2
        
        # Curriculum Learning
        self.min_interaction_rounds = 1
        self.max_interaction_rounds = 3
        
        self.prob_bbox = 0.6
        self.prob_points = 0.9
        self.prob_scribbles = 0.3
        
        self.stats = {'samples': 0, 'instances_per_sample': []}

    def set_instance_mapping(self, json_path: str):
        """Setze instance mapping nach Initialisierung."""
        self.instance_mapping_path = json_path
        if json_path and Path(json_path).exists():
            self._load_instance_mappings(json_path)


    def do_split(self):
        """Override: Handle small datasets (<5 samples)."""
        # Nutze die dataset_json um die Anzahl der Cases zu bestimmen
        dataset_json = self.dataset_json
        
        if 'numTraining' in dataset_json:
            num_samples = dataset_json['numTraining']
        else:
            # Fallback: Zähle Files im preprocessed Ordner
            from nnunetv2.paths import nnUNet_preprocessed
            preprocessed_folder = Path(nnUNet_preprocessed) / self.plans_manager.dataset_name
            all_keys = []
            for f in preprocessed_folder.glob('*_0000.npz'):
                case_id = f.name.replace('_0000.npz', '')
                all_keys.append(case_id)
            num_samples = len(all_keys)
        
        if num_samples < 5:
            print(f"⚠️  Only {num_samples} samples - using simple train/val split")
            # Lade die Keys aus dem preprocessed folder
            from nnunetv2.paths import nnUNet_preprocessed
            preprocessed_folder = Path(nnUNet_preprocessed) / self.plans_manager.dataset_name / \
                                 f'nnUNetPlans_{self.configuration_name}'
            
            all_keys = []
            for f in sorted(preprocessed_folder.glob('*.npz')):
                case_id = f.stem.replace('_0000', '')
                if case_id not in all_keys:
                    all_keys.append(case_id)
            
            # Nutze alle für Training und Validation
            return all_keys, all_keys
        
        # Normal CV für größere Datasets
        return super().do_split()

    def _load_instance_mappings(self, json_path: str):
        """Lade instance_to_material Mappings aus JSON."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for case_name, instances in data.items():
            case_key = case_name.replace('.nii', '')
            self.instance_mappings[case_key] = {}
            
            for inst_id_str, inst_info in instances.items():
                inst_id = int(inst_id_str)
                material = inst_info['material']
                
                if material.startswith('unknown_'):
                    material = self._map_unknown_material(inst_info['material_label'])
                
                self.instance_mappings[case_key][inst_id] = material
        
        print(f"Loaded instance mappings for {len(self.instance_mappings)} cases")

    def _map_unknown_material(self, material_label: int) -> str:
        """Map unknown material labels to known materials."""
        if material_label in self.label_to_material:
            return self.label_to_material[material_label]
        return 'aluminum'

    def initialize(self):
        if not self.was_initialized:
            configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
            self.batch_size = configuration_manager.batch_size
            self.patch_size = configuration_manager.patch_size
            
            self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            
            self.build_network_architecture()
            self._build_loss()
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            self.grad_scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
            
            self.was_initialized = True
            
            print("=" * 70)
            print("INTERACTIVE INSTANCE SEGMENTATION TRAINER - INITIALIZED")
            print("=" * 70)
            print(f"Input Channels:   1 (CT) + {self.num_materials} (Materials) + 7 (Interact) = {self.num_input_channels}")
            print(f"Output Classes:   {self.num_output_classes} (Background, Object)")
            print(f"Device:           {self.device}")
            print(f"Materials:        {list(self.material_to_label.keys())}")
            print(f"Instance Maps:    {len(self.instance_mappings)} cases loaded")
            print("=" * 70)

    def build_network_architecture(self):
        import torch.nn as nn
        from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
        
        configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
        network_kwargs = configuration_manager.network_arch_init_kwargs.copy()
        
        if isinstance(network_kwargs.get('conv_op'), str):
            network_kwargs['conv_op'] = nn.Conv3d if 'Conv3d' in network_kwargs['conv_op'] else nn.Conv2d
        if isinstance(network_kwargs.get('norm_op'), str):
            network_kwargs['norm_op'] = nn.InstanceNorm3d if 'InstanceNorm3d' in network_kwargs['norm_op'] else nn.InstanceNorm2d
        if isinstance(network_kwargs.get('nonlin'), str):
            network_kwargs['nonlin'] = nn.LeakyReLU
        
        network_kwargs['input_channels'] = self.num_input_channels
        network_kwargs['num_classes'] = self.num_output_classes
        
        arch_class_name = configuration_manager.network_arch_class_name
        network_class = PlainConvUNet if 'PlainConvUNet' in arch_class_name else ResidualEncoderUNet
        
        self.network = network_class(**network_kwargs)
        self.network.apply(InitWeights_He(1e-2))
        init_last_bn_before_add_to_0(self.network)
        self.network.to(self.device)

    def _build_loss(self):
        self.loss = DC_and_CE_loss(
            soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={},
            weight_ce=1.0, weight_dice=1.0, ignore_label=None
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), 
            lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: (1 - epoch / self.num_epochs) ** 0.9
        )
        return optimizer, lr_scheduler

    # ==================== INSTANCE EXTRACTION ====================
    
    def extract_instances_from_labels(self, label_map: np.ndarray, 
                                     case_identifier: str) -> List[Dict]:
        """Extrahiert alle Instanzen aus dem Label-Bild."""
        unique_labels = np.unique(label_map)
        instances = []
        
        case_mapping = self.instance_mappings.get(case_identifier, {})
        
        for inst_id in unique_labels:
            if inst_id == 0:
                continue
            
            mask = (label_map == inst_id).astype(np.uint8)
            material = case_mapping.get(int(inst_id), 'aluminum')
            
            if material == 'background' or material not in self.material_to_label:
                continue
            
            instances.append({
                'instance_id': int(inst_id),
                'material': material,
                'mask': mask
            })
        
        return instances

    def encode_material(self, material: str, shape: Tuple[int, ...]) -> np.ndarray:
        """One-Hot Encoding für Material-Typ."""
        material_map = np.zeros((self.num_materials, *shape), dtype=np.float32)
        
        if material in self.material_to_label:
            mat_label = self.material_to_label[material]
            if mat_label > 0:
                mat_idx = mat_label - 1
                material_map[mat_idx] = 1.0
        
        return material_map

    # ==================== INTERAKTIONS-ENCODING ====================
    
    def sample_points_from_mask(self, mask: np.ndarray, num_points: int, 
                                avoid_border: bool = True) -> List[Tuple[int, ...]]:
        if avoid_border:
            eroded = binary_dilation(~mask.astype(bool), iterations=3)
            valid_mask = mask & ~eroded
        else:
            valid_mask = mask
        
        coords = np.argwhere(valid_mask)
        if len(coords) == 0:
            coords = np.argwhere(mask)
        if len(coords) == 0:
            return []
        
        num_points = min(num_points, len(coords))
        indices = np.random.choice(len(coords), size=num_points, replace=False)
        return [tuple(coords[i]) for i in indices]

    def encode_point(self, point: Tuple[int, ...], shape: Tuple[int, ...], 
                    radius: int) -> np.ndarray:
        point_map = np.zeros(shape, dtype=np.float32)
        coords = np.ogrid[tuple(slice(0, s) for s in shape)]
        dist_sq = sum((c - p) ** 2 for c, p in zip(coords, point))
        point_map = np.exp(-dist_sq / (2 * (radius / 2.0) ** 2))
        return point_map

    def get_bounding_box(self, mask: np.ndarray, margin: int = 5) -> Optional[List[Tuple[int, int]]]:
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
        return [(max(0, coords[:, d].min() - margin), 
                 min(mask.shape[d], coords[:, d].max() + margin + 1)) 
                for d in range(mask.ndim)]

    def encode_bbox(self, bbox: List[Tuple[int, int]], shape: Tuple[int, ...]) -> np.ndarray:
        bbox_map = np.zeros(shape, dtype=np.float32)
        bbox_map[tuple(slice(s, e) for s, e in bbox)] = 1.0
        return bbox_map

    def create_scribble(self, mask: np.ndarray, num_strokes: int = 2) -> np.ndarray:
        scribble_map = np.zeros_like(mask, dtype=np.float32)
        coords = np.argwhere(mask)
        if len(coords) < 2:
            return scribble_map
        
        for _ in range(num_strokes):
            idx = np.random.choice(len(coords), size=min(2, len(coords)), replace=False)
            if len(idx) < 2:
                continue
            try:
                line_coords = line_nd(coords[idx[0]], coords[idx[1]], endpoint=True)
                scribble_map[line_coords] = 1.0
                if self.scribble_thickness > 1:
                    scribble_map = binary_dilation(
                        scribble_map, 
                        iterations=self.scribble_thickness // 2
                    ).astype(np.float32)
            except:
                continue
        return scribble_map

    def simulate_instance_interaction(self, instance_mask: np.ndarray, 
                                     prev_pred: Optional[np.ndarray], 
                                     round_num: int) -> np.ndarray:
        """Simuliert User-Interaktion für EINE Instanz."""
        interaction_maps = np.zeros((7, *instance_mask.shape), dtype=np.float32)
        
        if instance_mask.sum() == 0:
            return interaction_maps
        
        if round_num == 0:
            if random.random() < self.prob_bbox:
                bbox = self.get_bounding_box(instance_mask, margin=random.randint(5, 10))
                if bbox:
                    interaction_maps[1] = self.encode_bbox(bbox, instance_mask.shape)
            
            points = self.sample_points_from_mask(instance_mask, random.randint(2, 4))
            for p in points:
                interaction_maps[3] = np.maximum(
                    interaction_maps[3], 
                    self.encode_point(p, instance_mask.shape, self.point_radius)
                )
        else:
            pred_mask = (prev_pred > 0.5).astype(np.uint8) if prev_pred is not None else np.zeros_like(instance_mask)
            
            fn_mask = (instance_mask == 1) & (pred_mask == 0)
            if fn_mask.sum() > 0 and random.random() < self.prob_points:
                points = self.sample_points_from_mask(fn_mask, random.randint(1, 3), avoid_border=False)
                for p in points:
                    interaction_maps[3] = np.maximum(
                        interaction_maps[3], 
                        self.encode_point(p, instance_mask.shape, self.point_radius)
                    )
            
            fp_mask = (instance_mask == 0) & (pred_mask == 1)
            if fp_mask.sum() > 0 and random.random() < self.prob_points * 0.7:
                points = self.sample_points_from_mask(fp_mask, random.randint(1, 2), avoid_border=False)
                for p in points:
                    interaction_maps[4] = np.maximum(
                        interaction_maps[4], 
                        self.encode_point(p, instance_mask.shape, self.point_radius)
                    )
            
            if random.random() < self.prob_scribbles:
                interaction_maps[5] = self.create_scribble(instance_mask)
        
        return interaction_maps

    # ==================== TRAINING ====================

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        case_ids = batch.get('keys', [f'case_{i:04d}' for i in range(data.shape[0])])
        
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        max_rounds = min(self.max_interaction_rounds, 1 + self.current_epoch // 30)
        num_rounds = random.randint(self.min_interaction_rounds, max_rounds)
        
        total_loss = 0
        current_pred = None
        valid_samples = 0
        
        for round_num in range(num_rounds):
            batch_inputs = []
            batch_targets = []
            
            for b in range(data.shape[0]):
                label_map = target[b, 0].cpu().numpy()
                case_id = case_ids[b] if isinstance(case_ids, list) else f'case_{b:04d}'
                
                instances = self.extract_instances_from_labels(label_map, case_id)
                
                if len(instances) == 0:
                    continue
                
                selected_instance = random.choice(instances)
                instance_mask = selected_instance['mask']
                material = selected_instance['material']
                
                prev_pred_mask = None
                if current_pred is not None and valid_samples > b:
                    prev_pred_mask = current_pred[b, 1].cpu().numpy()
                
                material_encoding = self.encode_material(material, instance_mask.shape)
                interaction_maps = self.simulate_instance_interaction(
                    instance_mask, prev_pred_mask, round_num
                )
                
                ct_image = data[b, 0].cpu().numpy()
                full_input = np.concatenate([
                    ct_image[np.newaxis],
                    material_encoding,
                    interaction_maps
                ], axis=0)
                
                batch_inputs.append(full_input)
                batch_targets.append(instance_mask.astype(np.int64))
            
            if len(batch_inputs) == 0:
                continue
            
            valid_samples = len(batch_inputs)
            
            input_tensor = torch.from_numpy(np.stack(batch_inputs)).to(
                self.device, dtype=torch.float32
            )
            target_tensor = torch.from_numpy(np.stack(batch_targets)).to(
                self.device, dtype=torch.long
            )
            
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                pred_logits = self.network(input_tensor)
                
                if isinstance(pred_logits, (list, tuple)):
                    pred_logits = pred_logits[0]
                
                loss = self.loss(pred_logits, target_tensor)
            
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.detach()
            
            with torch.no_grad():
                current_pred = torch.softmax(pred_logits, dim=1)
        
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': total_loss / max(num_rounds, 1)}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        if self.current_epoch % 10 == 0:
            avg_loss = np.mean([o['loss'].item() for o in train_outputs])
            print(f"Epoch {self.current_epoch:03d} | Loss: {avg_loss:.4f}")

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()

        dataset_tr = nnUNetDataset(
            self.preprocessed_dataset_folder,
            tr_keys,
            self.plans_manager,
            self.dataset_json,
            self.configuration_name
        )

        dataset_val = nnUNetDataset(
            self.preprocessed_dataset_folder,
            val_keys,
            self.plans_manager,
            self.dataset_json,
            self.configuration_name
        )

        print(f"✅ Training cases: {tr_keys}")
        print(f"✅ Validation cases: {val_keys}")

        return dataset_tr, dataset_val


    # ==================== INFERENCE ====================

    def predict_single_instance(self, ct_image: np.ndarray, 
                               click_point: Tuple[int, ...], 
                               material: str) -> np.ndarray:
        """Segmentiere EINE Instanz basierend auf User-Click."""
        self.network.eval()
        shape = ct_image.shape
        
        material_encoding = self.encode_material(material, shape)
        
        interaction_maps = np.zeros((7, *shape), dtype=np.float32)
        interaction_maps[3] = self.encode_point(click_point, shape, self.point_radius)
        
        full_input = np.concatenate([
            ct_image[np.newaxis],
            material_encoding,
            interaction_maps
        ], axis=0)
        
        input_tensor = torch.from_numpy(full_input[np.newaxis]).to(
            self.device, dtype=torch.float32
        )
        
        with torch.no_grad():
            pred_logits = self.network(input_tensor)
            if isinstance(pred_logits, (list, tuple)):
                pred_logits = pred_logits[0]
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_mask = (pred_probs[0, 1] > 0.5).cpu().numpy()
        
        return pred_mask.astype(np.uint8)

    def predict_full_scene(self, ct_image: np.ndarray, 
                          click_points: List[Tuple[Tuple[int, ...], str]]) -> np.ndarray:
        """Segmentiere mehrere Instanzen via mehrere Clicks."""
        instance_map = np.zeros(ct_image.shape, dtype=np.int32)
        
        for instance_id, (point, material) in enumerate(click_points, start=1):
            mask = self.predict_single_instance(ct_image, point, material)
            instance_map[mask > 0] = instance_id
        
        return instance_map


class InteractiveInstanceSegmentationTrainer(nnUNetTrainer):
    """
    Click-basierte Instance Segmentation für industrielle CT-Scans.
    
    FIXED: instance_mapping_path wird nach super().__init__ gesetzt
    """

    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict = None, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json=dataset_json, device=device)
        
        self.use_b2nd = True
        # Material-Klassen
        self.material_to_label = {
            'background': 0,
            'aluminum': 1,
            'plastic': 2,
            'steel': 3,
            'air': 4
        }
        self.label_to_material = {v: k for k, v in self.material_to_label.items()}
        self.num_materials = len(self.material_to_label) - 1  # Ohne Background
        
        # Instance mapping - wird später gesetzt
        self.instance_mapping_path = None
        self.instance_mappings = {}
        
        # Interaktions-Parameter
        self.point_radius = 4
        self.scribble_thickness = 3
        
        # Input: 1 (CT) + 4 (Materials) + 7 (Interactions) = 12
        self.num_input_channels = 1 + self.num_materials + 7
        
        # Output: 2 Klassen (Background, Object)
        self.num_output_classes = 2
        
        # Curriculum Learning
        self.min_interaction_rounds = 1
        self.max_interaction_rounds = 3
        
        self.prob_bbox = 0.6
        self.prob_points = 0.9
        self.prob_scribbles = 0.3
        
        self.stats = {'samples': 0, 'instances_per_sample': []}

    def set_instance_mapping(self, json_path: str):
        """Setze instance mapping nach Initialisierung."""
        self.instance_mapping_path = json_path
        if json_path and Path(json_path).exists():
            self._load_instance_mappings(json_path)


    def do_split(self):
        """Override: Handle small datasets (<5 samples)."""
        # Nutze die dataset_json um die Anzahl der Cases zu bestimmen
        dataset_json = self.dataset_json
        
        if 'numTraining' in dataset_json:
            num_samples = dataset_json['numTraining']
        else:
            # Fallback: Zähle Files im preprocessed Ordner
            from nnunetv2.paths import nnUNet_preprocessed
            preprocessed_folder = Path(nnUNet_preprocessed) / self.plans_manager.dataset_name
            all_keys = []
            for f in preprocessed_folder.glob('*_0000.npz'):
                case_id = f.name.replace('_0000.npz', '')
                all_keys.append(case_id)
            num_samples = len(all_keys)
        
        if num_samples < 5:
            print(f"⚠️  Only {num_samples} samples - using simple train/val split")
            # Lade die Keys aus dem preprocessed folder
            from nnunetv2.paths import nnUNet_preprocessed
            preprocessed_folder = Path(nnUNet_preprocessed) / self.plans_manager.dataset_name / \
                                 f'nnUNetPlans_{self.configuration_name}'
            
            all_keys = []
            for f in sorted(preprocessed_folder.glob('*.npz')):
                case_id = f.stem.replace('_0000', '')
                if case_id not in all_keys:
                    all_keys.append(case_id)
            
            # Nutze alle für Training und Validation
            return all_keys, all_keys
        
        # Normal CV für größere Datasets
        return super().do_split()

    def _load_instance_mappings(self, json_path: str):
        """Lade instance_to_material Mappings aus JSON."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for case_name, instances in data.items():
            case_key = case_name.replace('.nii', '')
            self.instance_mappings[case_key] = {}
            
            for inst_id_str, inst_info in instances.items():
                inst_id = int(inst_id_str)
                material = inst_info['material']
                
                if material.startswith('unknown_'):
                    material = self._map_unknown_material(inst_info['material_label'])
                
                self.instance_mappings[case_key][inst_id] = material
        
        print(f"Loaded instance mappings for {len(self.instance_mappings)} cases")

    def _map_unknown_material(self, material_label: int) -> str:
        """Map unknown material labels to known materials."""
        if material_label in self.label_to_material:
            return self.label_to_material[material_label]
        return 'aluminum'

    def initialize(self):
        if not self.was_initialized:
            configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
            self.batch_size = configuration_manager.batch_size
            self.patch_size = configuration_manager.patch_size
            
            self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            
            self.build_network_architecture()
            self._build_loss()
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            self.grad_scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
            
            self.was_initialized = True
            
            print("=" * 70)
            print("INTERACTIVE INSTANCE SEGMENTATION TRAINER - INITIALIZED")
            print("=" * 70)
            print(f"Input Channels:   1 (CT) + {self.num_materials} (Materials) + 7 (Interact) = {self.num_input_channels}")
            print(f"Output Classes:   {self.num_output_classes} (Background, Object)")
            print(f"Device:           {self.device}")
            print(f"Materials:        {list(self.material_to_label.keys())}")
            print(f"Instance Maps:    {len(self.instance_mappings)} cases loaded")
            print("=" * 70)

    def build_network_architecture(self):
        import torch.nn as nn
        from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
        
        configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
        network_kwargs = configuration_manager.network_arch_init_kwargs.copy()
        
        if isinstance(network_kwargs.get('conv_op'), str):
            network_kwargs['conv_op'] = nn.Conv3d if 'Conv3d' in network_kwargs['conv_op'] else nn.Conv2d
        if isinstance(network_kwargs.get('norm_op'), str):
            network_kwargs['norm_op'] = nn.InstanceNorm3d if 'InstanceNorm3d' in network_kwargs['norm_op'] else nn.InstanceNorm2d
        if isinstance(network_kwargs.get('nonlin'), str):
            network_kwargs['nonlin'] = nn.LeakyReLU
        
        network_kwargs['input_channels'] = self.num_input_channels
        network_kwargs['num_classes'] = self.num_output_classes
        
        arch_class_name = configuration_manager.network_arch_class_name
        network_class = PlainConvUNet if 'PlainConvUNet' in arch_class_name else ResidualEncoderUNet
        
        self.network = network_class(**network_kwargs)
        self.network.apply(InitWeights_He(1e-2))
        init_last_bn_before_add_to_0(self.network)
        self.network.to(self.device)

    def _build_loss(self):
        self.loss = DC_and_CE_loss(
            soft_dice_kwargs={'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={},
            weight_ce=1.0, weight_dice=1.0, ignore_label=None
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), 
            lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda epoch: (1 - epoch / self.num_epochs) ** 0.9
        )
        return optimizer, lr_scheduler

    # ==================== INSTANCE EXTRACTION ====================
    
    def extract_instances_from_labels(self, label_map: np.ndarray, 
                                     case_identifier: str) -> List[Dict]:
        """Extrahiert alle Instanzen aus dem Label-Bild."""
        unique_labels = np.unique(label_map)
        instances = []
        
        case_mapping = self.instance_mappings.get(case_identifier, {})
        
        for inst_id in unique_labels:
            if inst_id == 0:
                continue
            
            mask = (label_map == inst_id).astype(np.uint8)
            material = case_mapping.get(int(inst_id), 'aluminum')
            
            if material == 'background' or material not in self.material_to_label:
                continue
            
            instances.append({
                'instance_id': int(inst_id),
                'material': material,
                'mask': mask
            })
        
        return instances

    def encode_material(self, material: str, shape: Tuple[int, ...]) -> np.ndarray:
        """One-Hot Encoding für Material-Typ."""
        material_map = np.zeros((self.num_materials, *shape), dtype=np.float32)
        
        if material in self.material_to_label:
            mat_label = self.material_to_label[material]
            if mat_label > 0:
                mat_idx = mat_label - 1
                material_map[mat_idx] = 1.0
        
        return material_map

    # ==================== INTERAKTIONS-ENCODING ====================
    
    def sample_points_from_mask(self, mask: np.ndarray, num_points: int, 
                                avoid_border: bool = True) -> List[Tuple[int, ...]]:
        if avoid_border:
            eroded = binary_dilation(~mask.astype(bool), iterations=3)
            valid_mask = mask & ~eroded
        else:
            valid_mask = mask
        
        coords = np.argwhere(valid_mask)
        if len(coords) == 0:
            coords = np.argwhere(mask)
        if len(coords) == 0:
            return []
        
        num_points = min(num_points, len(coords))
        indices = np.random.choice(len(coords), size=num_points, replace=False)
        return [tuple(coords[i]) for i in indices]

    def encode_point(self, point: Tuple[int, ...], shape: Tuple[int, ...], 
                    radius: int) -> np.ndarray:
        point_map = np.zeros(shape, dtype=np.float32)
        coords = np.ogrid[tuple(slice(0, s) for s in shape)]
        dist_sq = sum((c - p) ** 2 for c, p in zip(coords, point))
        point_map = np.exp(-dist_sq / (2 * (radius / 2.0) ** 2))
        return point_map

    def get_bounding_box(self, mask: np.ndarray, margin: int = 5) -> Optional[List[Tuple[int, int]]]:
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
        return [(max(0, coords[:, d].min() - margin), 
                 min(mask.shape[d], coords[:, d].max() + margin + 1)) 
                for d in range(mask.ndim)]

    def encode_bbox(self, bbox: List[Tuple[int, int]], shape: Tuple[int, ...]) -> np.ndarray:
        bbox_map = np.zeros(shape, dtype=np.float32)
        bbox_map[tuple(slice(s, e) for s, e in bbox)] = 1.0
        return bbox_map

    def create_scribble(self, mask: np.ndarray, num_strokes: int = 2) -> np.ndarray:
        scribble_map = np.zeros_like(mask, dtype=np.float32)
        coords = np.argwhere(mask)
        if len(coords) < 2:
            return scribble_map
        
        for _ in range(num_strokes):
            idx = np.random.choice(len(coords), size=min(2, len(coords)), replace=False)
            if len(idx) < 2:
                continue
            try:
                line_coords = line_nd(coords[idx[0]], coords[idx[1]], endpoint=True)
                scribble_map[line_coords] = 1.0
                if self.scribble_thickness > 1:
                    scribble_map = binary_dilation(
                        scribble_map, 
                        iterations=self.scribble_thickness // 2
                    ).astype(np.float32)
            except:
                continue
        return scribble_map

    def simulate_instance_interaction(self, instance_mask: np.ndarray, 
                                     prev_pred: Optional[np.ndarray], 
                                     round_num: int) -> np.ndarray:
        """Simuliert User-Interaktion für EINE Instanz."""
        interaction_maps = np.zeros((7, *instance_mask.shape), dtype=np.float32)
        
        if instance_mask.sum() == 0:
            return interaction_maps
        
        if round_num == 0:
            if random.random() < self.prob_bbox:
                bbox = self.get_bounding_box(instance_mask, margin=random.randint(5, 10))
                if bbox:
                    interaction_maps[1] = self.encode_bbox(bbox, instance_mask.shape)
            
            points = self.sample_points_from_mask(instance_mask, random.randint(2, 4))
            for p in points:
                interaction_maps[3] = np.maximum(
                    interaction_maps[3], 
                    self.encode_point(p, instance_mask.shape, self.point_radius)
                )
        else:
            pred_mask = (prev_pred > 0.5).astype(np.uint8) if prev_pred is not None else np.zeros_like(instance_mask)
            
            fn_mask = (instance_mask == 1) & (pred_mask == 0)
            if fn_mask.sum() > 0 and random.random() < self.prob_points:
                points = self.sample_points_from_mask(fn_mask, random.randint(1, 3), avoid_border=False)
                for p in points:
                    interaction_maps[3] = np.maximum(
                        interaction_maps[3], 
                        self.encode_point(p, instance_mask.shape, self.point_radius)
                    )
            
            fp_mask = (instance_mask == 0) & (pred_mask == 1)
            if fp_mask.sum() > 0 and random.random() < self.prob_points * 0.7:
                points = self.sample_points_from_mask(fp_mask, random.randint(1, 2), avoid_border=False)
                for p in points:
                    interaction_maps[4] = np.maximum(
                        interaction_maps[4], 
                        self.encode_point(p, instance_mask.shape, self.point_radius)
                    )
            
            if random.random() < self.prob_scribbles:
                interaction_maps[5] = self.create_scribble(instance_mask)
        
        return interaction_maps

    # ==================== TRAINING ====================

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        
        case_ids = batch.get('keys', [f'case_{i:04d}' for i in range(data.shape[0])])
        
        if isinstance(target, list):
            target = target[0]
        target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        max_rounds = min(self.max_interaction_rounds, 1 + self.current_epoch // 30)
        num_rounds = random.randint(self.min_interaction_rounds, max_rounds)
        
        total_loss = 0
        current_pred = None
        valid_samples = 0
        
        for round_num in range(num_rounds):
            batch_inputs = []
            batch_targets = []
            
            for b in range(data.shape[0]):
                label_map = target[b, 0].cpu().numpy()
                case_id = case_ids[b] if isinstance(case_ids, list) else f'case_{b:04d}'
                
                instances = self.extract_instances_from_labels(label_map, case_id)
                
                if len(instances) == 0:
                    continue
                
                selected_instance = random.choice(instances)
                instance_mask = selected_instance['mask']
                material = selected_instance['material']
                
                prev_pred_mask = None
                if current_pred is not None and valid_samples > b:
                    prev_pred_mask = current_pred[b, 1].cpu().numpy()
                
                material_encoding = self.encode_material(material, instance_mask.shape)
                interaction_maps = self.simulate_instance_interaction(
                    instance_mask, prev_pred_mask, round_num
                )
                
                ct_image = data[b, 0].cpu().numpy()
                full_input = np.concatenate([
                    ct_image[np.newaxis],
                    material_encoding,
                    interaction_maps
                ], axis=0)
                
                batch_inputs.append(full_input)
                batch_targets.append(instance_mask.astype(np.int64))
            
            if len(batch_inputs) == 0:
                continue
            
            valid_samples = len(batch_inputs)
            
            input_tensor = torch.from_numpy(np.stack(batch_inputs)).to(
                self.device, dtype=torch.float32
            )
            target_tensor = torch.from_numpy(np.stack(batch_targets)).to(
                self.device, dtype=torch.long
            )
            
            with torch.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                pred_logits = self.network(input_tensor)
                
                if isinstance(pred_logits, (list, tuple)):
                    pred_logits = pred_logits[0]
                
                loss = self.loss(pred_logits, target_tensor)
            
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.detach()
            
            with torch.no_grad():
                current_pred = torch.softmax(pred_logits, dim=1)
        
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': total_loss / max(num_rounds, 1)}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        if self.current_epoch % 10 == 0:
            avg_loss = np.mean([o['loss'].item() for o in train_outputs])
            print(f"Epoch {self.current_epoch:03d} | Loss: {avg_loss:.4f}")

    # ==================== INFERENCE ====================

    def predict_single_instance(self, ct_image: np.ndarray, 
                               click_point: Tuple[int, ...], 
                               material: str) -> np.ndarray:
        """Segmentiere EINE Instanz basierend auf User-Click."""
        self.network.eval()
        shape = ct_image.shape
        
        material_encoding = self.encode_material(material, shape)
        
        interaction_maps = np.zeros((7, *shape), dtype=np.float32)
        interaction_maps[3] = self.encode_point(click_point, shape, self.point_radius)
        
        full_input = np.concatenate([
            ct_image[np.newaxis],
            material_encoding,
            interaction_maps
        ], axis=0)
        
        input_tensor = torch.from_numpy(full_input[np.newaxis]).to(
            self.device, dtype=torch.float32
        )
        
        with torch.no_grad():
            pred_logits = self.network(input_tensor)
            if isinstance(pred_logits, (list, tuple)):
                pred_logits = pred_logits[0]
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_mask = (pred_probs[0, 1] > 0.5).cpu().numpy()
        
        return pred_mask.astype(np.uint8)

    def predict_full_scene(self, ct_image: np.ndarray, 
                          click_points: List[Tuple[Tuple[int, ...], str]]) -> np.ndarray:
        """Segmentiere mehrere Instanzen via mehrere Clicks."""
        instance_map = np.zeros(ct_image.shape, dtype=np.int32)
        
        for instance_id, (point, material) in enumerate(click_points, start=1):
            mask = self.predict_single_instance(ct_image, point, material)
            instance_map[mask > 0] = instance_id
        
        return instance_map