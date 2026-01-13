"""
Instance Label Preprocessor für Connected Component Segmentation
==================================================================

Dieses Script bereitet deine Labels vor für "Click-to-Segment" Training:
- Jedes zusammenhängende Teil bekommt eine eindeutige ID
- Material-Information wird separat gespeichert
- Für Interactive Training mit Connected Components

Use Case:
- Klick auf Aluminium-Teil A → nur Teil A wird segmentiert
- NICHT auch Teil B (auch wenn es Aluminium ist)
"""

import numpy as np
from scipy import ndimage
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import nibabel as nib


class InstanceLabelPreprocessor:
    """
    Konvertiert Material-Labels zu Instance-Labels für Connected Component Segmentation.
    """
    
    def __init__(self):
        self.instance_to_material = {}  # Maps instance_id → material_name
        self.material_to_label = {}     # Maps material_name → label_id
        
    def setup_materials(self, material_dict: Dict[str, int]):
        """
        Definiere Material-Mapping.
        
        Args:
            material_dict: z.B. {'aluminum': 1, 'plastic': 2, 'steel': 3}
        """
        self.material_to_label = {'background': 0, **material_dict}
        print(f"Defined {len(self.material_to_label)} materials:")
        for mat, label in self.material_to_label.items():
            print(f"  {label}: {mat}")
    
    def split_into_instances(self, 
                            label_array: np.ndarray,
                            connectivity: int = 2) -> Tuple[np.ndarray, Dict]:
        """
        Teile Material-Labels in einzelne zusammenhängende Instanzen auf.
        
        Args:
            label_array: Material-Labels [H, W, D]
                        z.B. 1=aluminum, 2=plastic, 3=steel
            connectivity: 1=face-connected, 2=edge-connected, 3=vertex-connected
                         Für 3D CT empfohlen: 2 (edge-connected)
        
        Returns:
            instance_labels: [H, W, D] mit eindeutigen Instance-IDs
            instance_info: Dict mit Metadaten
        
        Example:
            Input:  [1, 1, 0, 2, 2]  # 2x aluminum (getrennt), 1x plastic
                     [1, 0, 0, 2, 0]
                     
            Output: [1, 1, 0, 3, 3]  # Instance 1, Instance 3
                     [2, 0, 0, 3, 0]  # Instance 2 (anderes Alu-Teil!)
        """
        instance_labels = np.zeros_like(label_array, dtype=np.uint32)
        instance_info = {}
        
        current_instance_id = 1
        
        # Invertiere material_to_label für lookup
        label_to_material = {v: k for k, v in self.material_to_label.items()}
        
        print(f"\nSplitting materials into connected components...")
        print(f"Connectivity: {connectivity} ({'face' if connectivity==1 else 'edge' if connectivity==2 else 'vertex'})")
        
        # Für jedes Material separat
        for material_label in sorted(set(label_array.flatten())):
            if material_label == 0:  # Skip background
                continue
            
            material_name = label_to_material.get(material_label, f"unknown_{material_label}")
            
            # Maske für dieses Material
            material_mask = (label_array == material_label)
            
            if material_mask.sum() == 0:
                continue
            
            # Finde zusammenhängende Komponenten
            # Strukturelement für Konnektivität
            if connectivity == 1:
                structure = ndimage.generate_binary_structure(label_array.ndim, 1)
            elif connectivity == 2:
                structure = ndimage.generate_binary_structure(label_array.ndim, 2)
            else:
                structure = ndimage.generate_binary_structure(label_array.ndim, 3)
            
            labeled_components, num_components = ndimage.label(
                material_mask, 
                structure=structure
            )
            
            print(f"  Material '{material_name}' (label {material_label}): "
                  f"{num_components} separate components")
            
            # Zuweise neue Instance-IDs
            for component_idx in range(1, num_components + 1):
                component_mask = (labeled_components == component_idx)
                voxel_count = component_mask.sum()
                
                # Filtere sehr kleine Komponenten (Rauschen)
                if voxel_count < 10:  # Threshold anpassbar
                    continue
                
                # Zuweise Instance-ID
                instance_labels[component_mask] = current_instance_id
                
                # Speichere Metadaten
                instance_info[current_instance_id] = {
                    'material': material_name,
                    'material_label': int(material_label),
                    'voxel_count': int(voxel_count),
                    'component_idx': component_idx
                }
                
                # Speichere für Rückkonvertierung
                self.instance_to_material[current_instance_id] = material_name
                
                current_instance_id += 1
        
        total_instances = current_instance_id - 1
        print(f"\n✓ Created {total_instances} unique instances from materials")
        
        return instance_labels, instance_info
    
    def visualize_split(self, 
                       original_labels: np.ndarray,
                       instance_labels: np.ndarray,
                       slice_idx: Optional[int] = None):
        """
        Zeige Unterschied zwischen Material- und Instance-Labels.
        
        Args:
            original_labels: Original Material-Labels
            instance_labels: Neue Instance-Labels
            slice_idx: Welcher Slice visualisiert werden soll (None = Mitte)
        """
        if slice_idx is None:
            slice_idx = original_labels.shape[-1] // 2
        
        print(f"\n{'='*60}")
        print(f"Visualization of slice {slice_idx}")
        print(f"{'='*60}")
        
        # Extrahiere 2D Slice
        orig_slice = original_labels[..., slice_idx]
        inst_slice = instance_labels[..., slice_idx]
        
        print(f"\nOriginal Material Labels:")
        print(f"  Unique values: {np.unique(orig_slice)}")
        
        print(f"\nInstance Labels:")
        print(f"  Unique values: {np.unique(inst_slice)}")
        
        # Zähle Instanzen pro Material
        print(f"\nInstances per material:")
        for material, label in self.material_to_label.items():
            if material == 'background':
                continue
            
            # Zähle wie viele verschiedene Instance-IDs zu diesem Material gehören
            material_mask = (orig_slice == label)
            instances_in_this_material = np.unique(inst_slice[material_mask])
            instances_in_this_material = instances_in_this_material[instances_in_this_material > 0]
            
            print(f"  {material:15s}: {len(instances_in_this_material)} instances in this slice")
    
    def save_instance_mapping(self, output_path: str):
        """Speichere Instance→Material Mapping als JSON"""
        mapping = {
            'instance_to_material': {
                str(k): v for k, v in self.instance_to_material.items()
            },
            'material_to_label': self.material_to_label
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"\n✓ Saved instance mapping to: {output_path}")
    
    def batch_process(self,
                     label_files: List[Path],
                     output_folder: Path,
                     connectivity: int = 2,
                     save_info: bool = True):
        """
        Verarbeite mehrere Label-Files.
        
        Args:
            label_files: Liste von Label-File-Pfaden
            output_folder: Wo instance-labels gespeichert werden
            connectivity: Konnektivität für Connected Components
            save_info: Speichere Metadaten als JSON
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Batch Processing {len(label_files)} label files")
        print(f"{'='*60}\n")
        
        all_instance_info = {}
        
        for label_file in tqdm(label_files, desc="Processing labels"):
            # Lade Label
            if label_file.suffix == '.npy':
                original_labels = np.load(label_file)
            elif label_file.suffix == '.npz':
                data = np.load(label_file)
                original_labels = data['label'] if 'label' in data else data['arr_0']
            elif label_file.suffix in ['.nii', '.gz', '.nii.gz']:
                nii = nib.load(str(label_file))
                original_labels = nii.get_fdata().astype(np.uint8)
                affine = nii.affine
            else:
                print(f"  ⚠ Skipping unsupported format: {label_file}")
                continue

            # Konvertiere zu Instances
            instance_labels, instance_info = self.split_into_instances(
                original_labels,
                connectivity=connectivity
            )

            # Speichere als .npy
            output_file = output_folder / f"{label_file.stem}_instances.npy"
            np.save(output_file, instance_labels)

            # Speichere als NIfTI direkt hier in der Schleife
            out_nii = nib.Nifti1Image(instance_labels.astype(np.uint32), affine)
            nib.save(out_nii, output_file.with_suffix('.nii.gz'))

            # Metadaten
            all_instance_info[label_file.stem] = instance_info


        
        # Speichere globale Metadaten
        if save_info:
            info_file = output_folder / "instance_info.json"
            with open(info_file, 'w') as f:
                json.dump(all_instance_info, f, indent=2)
            
            mapping_file = output_folder / "instance_to_material_mapping.json"
            self.save_instance_mapping(str(mapping_file))
            
        if 'instance_labels' in locals():
            out_nii = nib.Nifti1Image(instance_labels.astype(np.uint32), affine)
            nib.save(out_nii, output_file.with_suffix('.nii.gz'))


        print(f"\n✓ Batch processing complete!")
        print(f"  Instance labels saved to: {output_folder}")
        print(f"  Total files processed: {len(label_files)}")


# ==================== EXAMPLE USAGE ====================

def example_usage():
    """Beispiel: Material-Labels zu Instance-Labels konvertieren"""
    
    # 1. Setup
    preprocessor = InstanceLabelPreprocessor()
    
    # 2. Definiere Materialien (wie sie in deinen Labels kodiert sind)
    preprocessor.setup_materials({
        'aluminum': 1,
        'plastic': 2,
        'steel': 3,
        'air': 4
    })
    
    # 3. Option A: Einzelne Datei
    print("\n" + "="*60)
    print("Processing single file")
    print("="*60)
    
    # Lade dein Label
    original_labels = np.load('path/to/your/label_001.npy')
    
    # Konvertiere zu Instances
    instance_labels, instance_info = preprocessor.split_into_instances(
        original_labels,
        connectivity=2  # 2=edge-connected (empfohlen für 3D)
    )
    
    # Visualisiere
    preprocessor.visualize_split(original_labels, instance_labels)
    
    # Speichere
    np.save('path/to/output/label_001_instances.npy', instance_labels)
    preprocessor.save_instance_mapping('path/to/output/instance_mapping.json')
    
    # 4. Option B: Batch-Verarbeitung
    print("\n" + "="*60)
    print("Batch processing all labels")
    print("="*60)
    
    label_folder = Path("path/to/your/labels")
    output_folder = Path("path/to/output/instance_labels")
    
    label_files = sorted(label_folder.glob("*.npy"))
    
    preprocessor.batch_process(
        label_files=label_files,
        output_folder=output_folder,
        connectivity=2,
        save_info=True
    )
    
    print("\n" + "="*60)
    print("✓ All done! Next steps:")
    print("="*60)
    print("1. Use the instance labels with the dataset converter")
    print("2. Train with InteractiveNnUNetTrainer in 'instance' mode")
    print("3. Deploy: click on part → segment only that connected part")


def quick_test():
    """Schneller Test mit synthetischen Daten"""
    print("Creating test data...")
    
    # Erstelle Test-Label mit 2 Aluminium-Teilen (getrennt)
    test_label = np.zeros((50, 50, 30), dtype=np.uint8)
    
    # Aluminium-Teil 1 (links)
    test_label[10:20, 10:20, 10:20] = 1
    
    # Aluminium-Teil 2 (rechts, GETRENNT!)
    test_label[30:40, 30:40, 10:20] = 1
    
    # Kunststoff-Teil
    test_label[20:30, 20:30, 15:25] = 2
    
    # Stahl-Teil
    test_label[15:25, 35:45, 5:15] = 3
    
    print(f"Test label shape: {test_label.shape}")
    print(f"Materials present: {np.unique(test_label)}")
    
    # Verarbeite
    preprocessor = InstanceLabelPreprocessor()
    preprocessor.setup_materials({
        'aluminum': 1,
        'plastic': 2,
        'steel': 3
    })
    
    instance_labels, info = preprocessor.split_into_instances(test_label)
    
    print(f"\nResult:")
    print(f"  Original had 3 material types")
    print(f"  Split into {len(info)} instances:")
    for inst_id, inst_info in info.items():
        print(f"    Instance {inst_id}: {inst_info['material']} "
              f"({inst_info['voxel_count']} voxels)")
    
    # Visualisiere einen Slice
    preprocessor.visualize_split(test_label, instance_labels, slice_idx=15)


if __name__ == '__main__':
    from pathlib import Path
    label_folder = Path("/zhome/hollms/TrainerTest/Dataset/Dataset001_CT_Scans/labelsTr")
    output_folder = Path("/zhome/hollms/TrainerTest/Dataset/Dataset001_CT_Scans/instancesTr")

    preprocessor = InstanceLabelPreprocessor()
    preprocessor.setup_materials({
        'aluminum': 1,
        'plastic': 2,
        'steel': 3,
        'air': 4
    })

    label_files = list(label_folder.glob("*.nii.gz"))

    preprocessor.batch_process(
        label_files=label_files,
        output_folder=output_folder,
        connectivity=2,
        save_info=True
    )