import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def load_images(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    print(f"Fichiers dans {folder} :", files)
    images = []
    for f in files:
        data = np.load(os.path.join(folder, f))
        print(f"Chargement {f} shape:", data.shape)
        images.append(data)
    return images, files

# Charger images
reconstructions, rec_files = load_images('dataset/test_reconstructions')
ground_truths, gt_files = load_images('dataset/test_groundtruth')

if len(ground_truths) == 0 or len(reconstructions) == 0:
    print("Erreur : un des dossiers est vide ou les fichiers sont invalides.")
    exit()

print(f"📦 {len(ground_truths)} Ground Truths | {len(reconstructions)} Reconstructions")

psnr_scores = []
ssim_scores = []

for gt, rec in zip(ground_truths, reconstructions):
    # Normaliser les images sur [0,1] si valeurs >1 (ex: 0-255)
    gt_norm = gt.astype(np.float32)
    rec_norm = rec.astype(np.float32)

    # Détection de l'échelle des données pour normaliser
    max_val = max(gt_norm.max(), rec_norm.max())
    if max_val > 1.0:
        gt_norm /= 255.0
        rec_norm /= 255.0

    mse = np.mean((gt_norm - rec_norm) ** 2)
    print(f"MSE: {mse:.8f}")

    if mse == 0:
        psnr_val = float('inf')
    else:
        psnr_val = peak_signal_noise_ratio(gt_norm, rec_norm, data_range=1.0)

    ssim_val = structural_similarity(gt_norm, rec_norm, data_range=1.0)

    psnr_scores.append(psnr_val)
    ssim_scores.append(ssim_val)

# Moyennes (en ignorant inf dans PSNR pour moyenne)
finite_psnr = [v for v in psnr_scores if np.isfinite(v)]
mean_psnr = np.mean(finite_psnr) if finite_psnr else float('inf')
mean_ssim = np.mean(ssim_scores)

print(f"📊 PSNR moyen: {'inf' if mean_psnr == float('inf') else f'{mean_psnr:.2f}'}")
print(f"📊 SSIM moyen: {mean_ssim:.4f}")

# Affichage comparatif d’une image (index 0)
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

plt.show()
