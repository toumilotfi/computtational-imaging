import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Dossiers
gt_folder = "dataset/test_groundtruth"
rec_folder = "dataset/test_reconstructions"

# Lister et trier les fichiers .npy
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.npy')])
rec_files = sorted([f for f in os.listdir(rec_folder) if f.endswith('.npy')])

print(f"🔍 Ground truths : {len(gt_files)} fichiers")
print(f"🔍 Reconstructions : {len(rec_files)} fichiers")

# Vérification
assert len(gt_files) == len(rec_files), "⚠️ Les nombres de fichiers GT et Recon ne correspondent pas."

# Stocker les scores
psnr_scores = []
ssim_scores = []

for gt_name, rec_name in zip(gt_files, rec_files):
    gt_path = os.path.join(gt_folder, gt_name)
    rec_path = os.path.join(rec_folder, rec_name)

    # Charger les tableaux numpy
    gt = np.load(gt_path)
    rec = np.load(rec_path)

    # Normaliser si nécessaire
    if gt.max() > 1.0:
        gt = gt / 255.0
    if rec.max() > 1.0:
        rec = rec / 255.0

    # Calculer métriques
    psnr_val = psnr(gt, rec, data_range=1.0)
    ssim_val = ssim(gt, rec, data_range=1.0)

    psnr_scores.append(psnr_val)
    ssim_scores.append(ssim_val)

    print(f"\n📂 Fichier : {gt_name}")
    print(f"   PSNR : {psnr_val:.4f}")
    print(f"   SSIM : {ssim_val:.4f}")

    # Affichage
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(gt, cmap='gray')
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(rec, cmap='gray')
    axs[1].set_title("Reconstruction")
    axs[1].axis("off")

    plt.suptitle(gt_name)
    plt.tight_layout()
    plt.show()

# Moyennes globales
print("\n📊 Résultats globaux :")
print(f"PSNR moyen : {np.mean(psnr_scores):.4f}")
print(f"SSIM moyen : {np.mean(ssim_scores):.4f}")

   
