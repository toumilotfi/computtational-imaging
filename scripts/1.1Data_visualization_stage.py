import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_npy_images(folder, n=5):
    if not os.path.exists(folder):
        print(f"Erreur : le dossier {folder} n'existe pas.")
        return
    
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if not files:
        print(f"Aucun fichier .npy trouvé dans {folder}.")
        return
    
    files = files[:n]
    plt.figure(figsize=(15, 3))
    for i, fname in enumerate(files):
        img = np.load(os.path.join(folder, fname))
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(fname)
        plt.axis('off')
    plt.show()

visualize_npy_images('dataset/train_preprocessed', n=5)
