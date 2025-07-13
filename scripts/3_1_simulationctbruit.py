import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
import numpy as np
import os
# === Étape 1 : générer une image (ex : fantôme de Shepp-Logan) ===
image = shepp_logan_phantom()
image = resize(image, (128, 128))  # redimensionner

# === Étape 2 : générer un sinogramme ===
theta = np.linspace(0., 180., 60, endpoint=False)  # 60 angles uniformément répartis
sin = radon(image, theta=theta, circle=True)  # <- sin est bien défini ici

# === Étape 3 : fonction pour ajouter du bruit gaussien ===
def add_gaussian_noise(sinogram, noise_level=0.01):
    std = noise_level * np.max(sinogram)
    noise = np.random.normal(0, std, sinogram.shape)
    return sinogram + noise

# === Étape 4 : appliquer le bruit ===
noisy_sino = add_gaussian_noise(sin, noise_level=0.01)

# === Étape 5 : visualiser ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Sinogramme propre")
plt.imshow(sin, cmap='gray', aspect='auto')

plt.subplot(1, 2, 2)
plt.title("Sinogramme bruité (1%)")
plt.imshow(noisy_sino, cmap='gray', aspect='auto')

plt.tight_layout()


# Chemin de sortie
output_dir = 'dataset/sinograms_noisy'
os.makedirs(output_dir, exist_ok=True)

# Exemple : sauvegarde
np.save(os.path.join(output_dir, 'sample.npy'), noisy_sino)

plt.show()
