import os
import numpy as np

# Crée les dossiers s'ils n'existent pas déjà
os.makedirs('dataset/test_groundtruth', exist_ok=True)
os.makedirs('dataset/test_reconstructions', exist_ok=True)

# Crée une fausse image (matrice 64x64) pour ground truth
ground_truth = np.ones((64, 64)) * 100

# Crée une fausse image (matrice 64x64) pour reconstruction
reconstruction = np.ones((64, 64)) * 90

# Sauvegarde ces images dans les dossiers correspondants
np.save('dataset/test_groundtruth/image01.npy', ground_truth)
np.save('dataset/test_reconstructions/image01.npy', reconstruction)

print("Fichiers de test créés avec succès.")
