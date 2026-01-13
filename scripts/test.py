# Test-Script
try:
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import zoom
    from skimage.draw import line_nd
    import torch
    print("✓ All core dependencies installed!")
    
    # Optional checks
    try:
        import pydicom
        print("✓ pydicom available (DICOM support enabled)")
    except:
        print("⚠ pydicom not installed (DICOM support disabled)")
        
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstall with:")
    print("pip install nibabel scipy scikit-image tqdm pillow nnunetv2")

# check_dataset.py
from pathlib import Path

dataset_folder = Path("Dataset/Dataset_0001")
images = sorted((dataset_folder / "imagesTr").glob("*.nii.gz"))
labels = sorted((dataset_folder / "labelsTr").glob("*.nii.gz"))

print(f"Images: {len(images)}")
print(f"Labels: {len(labels)}")

# Zeige erste 3
print("\nFirst 3 images:")
for img in images[:3]:
    print(f"  {img.name}")

print("\nFirst 3 labels:")
for lbl in labels[:3]:
    print(f"  {lbl.name}")

# Check matching
print("\nChecking if images have matching labels...")
for img in images[:5]:
    case_id = img.stem.replace("_0000", "")  # case_0000_0000 → case_0000
    expected_label = dataset_folder / "labelsTr" / f"{case_id}.nii.gz"
    
    if expected_label.exists():
        print(f"  ✓ {img.name} has label")
    else:
        print(f"  ✗ {img.name} MISSING label: {expected_label.name}")