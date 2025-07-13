import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

print(os.listdir('dataset/test_groundtruth'))
print(os.listdir('dataset/test_reconstructions'))
os.makedirs('dataset/test_groundtruth', exist_ok=True)

def load_images(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    print(f"Fichiers dans {folder} :", files)
    images = []
    for f in files:
        data = np.load(os.path.join(folder, f))
        print(f"Chargement {f} shape:", data.shape)
        images.append(data)
    return images, files
# Charger images reconstruites et ground-truth test
reconstructions, files = load_images('dataset/test_reconstructions')  # images sorties du modèle RISING
ground_truths, _ = load_images('dataset/test_groundtruth')

psnr_scores = []
ssim_scores = []

for rec, gt in zip(reconstructions, ground_truths):
    psnr_scores.append(psnr(gt, rec, data_range=1))
    ssim_scores.append(ssim(gt, rec, data_range=1))

print(f"PSNR moyen: {np.mean(psnr_scores):.2f}")
print(f"SSIM moyen: {np.mean(ssim_scores):.4f}")
gt_path = "dataset/test_groundtruth/image01.npy"  # adapte le chemin si nécessaire

gt = np.load(gt_path)

# === Affichage des infos de l'image ===
print("Min / Max de l'image ground-truth :", np.min(gt), np.max(gt))

# === Affichage de l'image ===
plt.imshow(gt, cmap='gray')
plt.title("Ground Truth")
plt.colorbar()
# Affichage comparatif d’une image au hasard
idx = 0
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Ground Truth')
plt.imshow(ground_truths[idx], cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title('Reconstruction')
plt.imshow(reconstructions[idx], cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Différence')
plt.imshow(np.abs(ground_truths[idx] - reconstructions[idx]), cmap='hot')
plt.axis('off')
ground_truths, gt_files = load_images('dataset/test_groundtruth')
reconstructions, rec_files = load_images('dataset/test_reconstructions')

print(f"Ground truths: {len(ground_truths)} fichiers")
print(f"Reconstructions: {len(reconstructions)} fichiers")

if len(ground_truths) == 0 or len(reconstructions) == 0:
    print("Erreur : un des dossiers est vide ou les fichiers sont invalides.")
    exit()

# Test shapes
print("Shape ground_truths[0]:", ground_truths[0].shape)
print("Shape reconstructions[0]:", reconstructions[0].shape)

# Ensuite calculs PSNR, SSIM etc...
print("GT min/max:", ground_truths[0].min(), ground_truths[0].max())
print("Rec min/max:", reconstructions[0].min(), reconstructions[0].max())
gt = ground_truths[0].astype(np.float32) / 255.0
rec = reconstructions[0].astype(np.float32) / 255.0
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

psnr = peak_signal_noise_ratio(gt, rec, data_range=1.0)
ssim = structural_similarity(gt, rec, data_range=1.0)

print(f"PSNR: {psnr:.4f}")
print(f"SSIM: {ssim:.4f}")


plt.show()
