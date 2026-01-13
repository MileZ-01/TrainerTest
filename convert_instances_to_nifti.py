
#!/usr/bin/env python3
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# Pfade
instances_dir = Path("/zhome/hollms/TrainerTest/Dataset/Dataset001_CT_Scans/labelsTr/instancesTr")
output_dir = Path("/zhome/hollms/TrainerTest/Dataset/Dataset001_CT_Scans/labelsTr")
images_dir = Path("/zhome/hollms/TrainerTest/Dataset/Dataset001_CT_Scans/imagesTr")

output_dir.mkdir(exist_ok=True)

npy_files = sorted(instances_dir.glob("*.npy"))
print(f"Found {len(npy_files)} instance label files")

for npy_file in tqdm(npy_files):
    # case_0000.nii_instances.npy -> case_0000
    case_name = npy_file.name.replace("_instances.npy", "").replace(".nii", "")
    
    # Lade instance labels
    instances = np.load(npy_file)
    print(f"\n{case_name}: Shape={instances.shape}, Unique labels={np.unique(instances)}")
    
    # Finde CT-Bild
    ct_file = images_dir / f"{case_name}_0000.nii.gz"
    
    if ct_file.exists():
        ct_img = nib.load(ct_file)
        affine = ct_img.affine
        header = ct_img.header.copy()
    else:
        print(f"⚠️  CT image not found: {ct_file}")
        affine = np.eye(4)
        header = None
    
    # Erstelle NIfTI mit Instance-Labels
    instances_nii = nib.Nifti1Image(instances.astype(np.int16), affine, header)
    
    # Speichere direkt in labelsTr/
    output_file = output_dir / f"{case_name}.nii.gz"
    nib.save(instances_nii, output_file)

print(f"\n✅ Converted {len(npy_files)} files to: {output_dir}")
print("Ready for nnU-Net preprocessing!")

