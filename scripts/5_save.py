import os
import numpy as np
from skimage.transform import radon, iradon
import numpy as np

  # Assure-toi que ce fichier existe et est dans le même dossier
import numpy as np
#from algo_iteratif import chambolle_pock
from algo_iteratif4 import chambolle_pock;
# Paramètres
angles = np.linspace(0, 180, 60, endpoint=False)  # 60 angles uniformes
lambd = 0.1
k_iter = 10
max_iter = 50
img_size = 128

# Dossiers
input_folder = "dataset/train_preprocessed"
output_folder_xk = "dataset/train_reconstructions/x_k"
output_folder_xstar = "dataset/train_reconstructions/x_star"
os.makedirs(output_folder_xk, exist_ok=True)
os.makedirs(output_folder_xstar, exist_ok=True)

def K(x):
    return radon(x, theta=angles, circle=True)

def KT(y):
    return iradon(y, theta=angles, filter_name=None, circle=True, output_size=img_size)


def create_reconstructions(img_path):
    x_true = np.load(img_path)  # image ground-truth
    y = K(x_true)  # sinogramme sans bruit
    noise = 0.01 * np.random.randn(*y.shape)  # bruit 1%
    y_delta = y + noise

    # Reconstruction complète à convergence
    x_star = chambolle_pock(y_delta, K, KT, lambd, max_iter=max_iter)

    # Reconstruction partielle à k_iter (modifie chambolle_pock pour sortir aussi x_k)
    x_k = chambolle_pock(y_delta, K, KT, lambd, max_iter=k_iter)

    return x_k, x_star

# Traitement sur tout le dataset train
for fname in os.listdir(input_folder):
    if fname.endswith('.npy'):
        path = os.path.join(input_folder, fname)
        x_k, x_star = create_reconstructions(path)

        base = os.path.splitext(fname)[0]
        np.save(os.path.join(output_folder_xk, base + ".npy"), x_k)
        np.save(os.path.join(output_folder_xstar, base + ".npy"), x_star)

print("Reconstructions enregistrées.")
