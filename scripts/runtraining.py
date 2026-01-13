#!/usr/bin/env python3
"""
Training Script für Interactive Instance Segmentation
"""
import os
import torch
from pathlib import Path

# nnU-Net Imports
from nnunetv2.run.run_training import run_training

# Dein Custom Trainer
from nnunetv2.training.nnUNetTrainer.InteractiveInstanceSegmentationTrainer import (
    InteractiveInstanceSegmentationTrainer
)

def main():
    """
    Startet das Training mit deinem Custom Trainer.
    """
    
    # ==================== KONFIGURATION ====================
    
    # Dataset ID (z.B. Dataset001_YourData)
    dataset_name_or_id = "Dataset001_CT_Scans"
    
    # Konfiguration: '2d', '3d_fullres', '3d_lowres', etc.
    configuration = "3d_fullres"
    
    # Fold (0-4 für 5-Fold Cross-Validation, oder 'all')
    fold = 0
    
    # Pfad zu deiner Instance-Mapping JSON
    instance_mapping_path = "/lgrp/edu-2025-2-brprj-segmentation/data/instance_metadata.json"
    
    # GPU Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optional: Pretrained Weights (None für Training from scratch)
    pretrained_weights = None
    
    # nnU-Net Umgebungsvariablen (WICHTIG!)
    # PASSE DIESE PFADE AN DEINE STRUKTUR AN!
    os.environ['nnUNet_raw'] = "/lgrp/edu-2025-2-brprj-segmentation/nnUNet_raw"
    os.environ['nnUNet_preprocessed'] = "/lgrp/edu-2025-2-brprj-segmentation/nnUNet_preprocessed"
    os.environ['nnUNet_results'] = "/lgrp/edu-2025-2-brprj-segmentation/nnUNet_results"
    
    print("=" * 70)
    print("INTERACTIVE INSTANCE SEGMENTATION - TRAINING START")
    print("=" * 70)
    print(f"Dataset:              {dataset_name_or_id}")
    print(f"Configuration:        {configuration}")
    print(f"Fold:                 {fold}")
    print(f"Device:               {device}")
    print(f"Instance Mapping:     {instance_mapping_path}")
    print(f"nnUNet_raw:           {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed:  {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results:       {os.environ['nnUNet_results']}")
    print("=" * 70)
    
    # Prüfe ob Instance Mapping existiert
    if not Path(instance_mapping_path).exists():
        print(f"⚠️  WARNING: Instance mapping file not found: {instance_mapping_path}")
        print("    Training will continue but material detection may not work correctly!")
    
    # ==================== TRAINING STARTEN ====================
    
    # Methode 1: Via run_training (Standard nnU-Net Weg)
    # PROBLEM: run_training akzeptiert keine extra Argumente wie instance_mapping_path
    # LÖSUNG: Wir müssen den Trainer manuell initialisieren
    
    # Methode 2: Manuell (EMPFOHLEN)
    manual_training(
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        fold=fold,
        instance_mapping_path=instance_mapping_path,
        device=device,
        pretrained_weights=pretrained_weights
    )


def manual_training(dataset_name_or_id: str, configuration: str, fold: int,
                   instance_mapping_path: str, device: torch.device,
                   pretrained_weights: str = None):
    """
    Manuelles Training Setup (mit zusätzlichen Parametern).
    """
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
    import json
    from nnunetv2.paths import nnUNet_results    
    # 1. Plans laden
    preprocessed_dataset_folder = Path(nnUNet_preprocessed) / dataset_name_or_id
    plans_file = preprocessed_dataset_folder / "nnUNetPlans.json"
    
    if not plans_file.exists():
        raise FileNotFoundError(
            f"Plans file not found: {plans_file}\n"
            f"Did you run preprocessing?\n"
            f"Run: nnUNetv2_plan_and_preprocess -d {dataset_name_or_id}"
        )
    
    with open(plans_file, 'r') as f:
        plans = json.load(f)
    
    # 2. Dataset JSON laden
    dataset_json_file = preprocessed_dataset_folder / "dataset.json"
    if not dataset_json_file.exists():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_file}")
    
    with open(dataset_json_file, 'r') as f:
        dataset_json = json.load(f)
    
    # 3. Output Folder
    output_folder = Path(nnUNet_results) / dataset_name_or_id / \
               f"InteractiveInstanceSegmentationTrainer__nnUNetPlans__{configuration}"
    
    if fold != 'all':
        output_folder = output_folder / f"fold_{fold}"
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output folder: {output_folder}\n")
    
    # 4. Trainer initialisieren
    trainer = InteractiveInstanceSegmentationTrainer(
        plans=plans,
        configuration=configuration,
        fold=0,
        dataset_json=dataset_json,
        device=device
    )
    
    # 5. Initialisieren
    trainer.initialize()
    
    trainer.instance_mapping_path = instance_mapping_path
    if instance_mapping_path and Path(instance_mapping_path).exists():
        trainer._load_instance_mappings(instance_mapping_path)

    # 6. Optional: Pretrained Weights laden
    if pretrained_weights and Path(pretrained_weights).exists():
        print(f"Loading pretrained weights from: {pretrained_weights}")
        trainer.load_checkpoint(pretrained_weights)
    
    # 7. Training starten
    print("\n🚀 Starting training...\n")
    trainer.run_training()
    
    print("\n✅ Training completed!")
    print(f"📁 Results saved to: {output_folder}")


def via_command_line():
    """
    Alternative: Via nnU-Net Command Line (einfacher, aber keine custom args).
    """
    # Dieser Weg funktioniert NUR wenn instance_mapping_path hart-kodiert ist
    # oder via Umgebungsvariable übergeben wird
    
    # Terminal Command:
    command = """
    nnUNetv2_train Dataset001_IndustrialCT 3d_fullres 0 \
        -tr InteractiveInstanceSegmentationTrainer \
        --npz
    """
    print("\nAlternativ via Command Line:")
    print(command)
    print("\nAber dann musst du instance_mapping_path im Code hart-kodieren!")


if __name__ == "__main__":
    main()
