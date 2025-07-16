import numpy as np
from skimage.transform import radon, iradon
from scipy.ndimage import sobel

def tv_gradient(x):
    """Calcul du gradient de la Total Variation isotrope"""
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    grad_norm = np.sqrt(dx**2 + dy**2 + 1e-8)
    dx /= grad_norm
    dy /= grad_norm
    div = (dx - np.roll(dx, 1, axis=1)) + (dy - np.roll(dy, 1, axis=0))
    return div

def chambolle_pock(y_delta, K, KT, lambd, tau=0.01, sigma=0.01, theta=1.0, max_iter=50):
    """
    Reconstruction via Chambolle-Pock algorithm.
    
    Args:
        y_delta: sinogramme bruité
        K: fonction projection Radon
        KT: fonction rétroprojection (inversion)
        lambd: poids régularisation TV
        tau, sigma, theta: paramètres algorithme
        max_iter: nombre d’itérations
    
    Returns:
        x: image reconstruite
    """
    # initialisation
    x = KT(y_delta)  # reconstruction initiale (ex. backprojection)
    x_bar = x.copy()
    p = np.zeros_like(y_delta)  # variable duale
    
    for i in range(max_iter):
        # gradient ascent dual
        p += sigma * (K(x_bar) - y_delta)
        p /= np.maximum(1, np.abs(p))
        
        x_old = x.copy()
        # gradient descent primal + TV prox
        x -= tau * KT(p)
        x += tau * lambd * tv_gradient(x)
        
        x_bar = x + theta * (x - x_old)
        
        if i % 10 == 0:
            print(f"Iteration {i}")
    return x


# Fonctions K et KT avec skimage radon et iradon

def K(x, angles):
    return radon(x, theta=angles, circle=True)

def KT(y, angles, output_size=128):
    return iradon(y, theta=angles, filter_name=None, circle=True, output_size=output_size)


# Exemple d'utilisation
if __name__ == "__main__":
    angles = np.linspace(0, 180, 60, endpoint=False)
    y_delta = np.load("dataset/sinograms_noisy/sample.npy")  # sinogramme bruité
    x_rec = chambolle_pock(y_delta, lambda img: K(img, angles), lambda sino: KT(sino, angles), lambd=0.1, max_iter=50)
    
    import matplotlib.pyplot as plt
    plt.imshow(x_rec, cmap='gray')
    plt.title("Image reconstruite - Chambolle-Pock")
    plt.axis('off')
    plt.show()
