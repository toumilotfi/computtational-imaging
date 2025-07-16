import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

# === PARAMÈTRES ===
input_dirs = {
    'train': 'dataset/train/C002',
    'test': 'dataset/test/C081'
}
output_dirs = {
    'train': 'dataset/train_preprocessed',
    'test': 'dataset/test_preprocessed'
}
img_size = 128  # ou 256 selon ta config

# === CRÉER DOSSIERS DE SORTIE SI NÉCESSAIRE ===
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# === FONCTION DE NORMALISATION ===
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

# === FONCTION DE TRAITEMENT D’UNE IMAGE ===
def process_and_save_image(path, out_path):
    img = imread(path, as_gray=True)  # niveau de gris
    img_resized = resize(img, (img_size, img_size), anti_aliasing=True)
    img_norm = normalize(img_resized)

    np.save(out_path, img_norm)  # Sauvegarde au format .npy

# === TRAITEMENT DE TOUS LES hh g FICHIERS ===
for split in ['train', 'test']:
    print(f"\n🔄 Prétraitement des images: {split}")
    input_dir = input_dirs[split]
    output_dir = output_dirs[split]

    for fname in tqdm(os.listdir(input_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, fname)
            base_name = os.path.splitext(fname)[0]
            output_path = os.path.join(output_dir, base_name + ".npy")

            process_and_save_image(input_path, output_path)

print("\n✅ Prétraitement terminé.")
