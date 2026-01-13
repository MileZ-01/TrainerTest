import napari
import numpy as np
import nibabel as nib
from pathlib import Path

def load_label_file(file_path: Path):
    """
    Lädt eine Label-Datei: .npy oder .nii/.nii.gz
    """
    if file_path.suffix == '.npy':
        data = np.load(file_path)
        affine = None
    elif file_path.suffix == '.nii' or file_path.suffixes == ['.nii', '.gz']:
        nii = nib.load(str(file_path))
        data = nii.get_fdata().astype(np.uint32)
        affine = nii.affine
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return data, affine

def visualize_labels(original_label_path: str, instance_label_path: str):
    """
    Zeigt Material-Labels und Instance-Labels nebeneinander in Napari
    """
    original_path = Path(original_label_path)
    instance_path = Path(instance_label_path)

    # Lade Daten
    original_labels, _ = load_label_file(original_path)
    instance_labels, _ = load_label_file(instance_path)

    # Starte Napari Viewer
    viewer = napari.Viewer()

    # Material-Labels Layer hinzufügen
    mat_layer = viewer.add_labels(
        original_labels,
        name='Material Labels',
        opacity=0.5
    )

    # Farben manuell setzen: 0=background, 1=aluminum, 2=plastic, 3=steel, 4=air
    mat_layer.color = [
        'black',   # 0=background
        'red',     # 1=aluminum
        'green',   # 2=plastic
        'blue',    # 3=steel
        'yellow'   # 4=air
    ]

    # Instance-Labels Layer hinzufügen
    inst_layer = viewer.add_labels(
        instance_labels,
        name='Instance Labels',
        opacity=0.8
    )

    # Zufällige Farben für jede Instanz
    inst_layer.color = 'random'

    # Napari starten
    napari.run()


if __name__ == '__main__':
    # Beispiel-Pfade anpassen:
    original_file = r"C:\Users\hollm\OneDrive\Desktop\Forschungsprojekt_Train\Dataset\Dataset_0001\labelsTr\case_0000.nii.gz"
    instance_file = r"C:\Users\hollm\OneDrive\Desktop\Forschungsprojekt_Train\Dataset\Dataset_0001\instancesTr\case_0000.nii_instances.npy"

    visualize_labels(original_file, instance_file)

