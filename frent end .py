import tkinter as tk
from tkinter import Image, filedialog, messagebox

def ouvrir_image():
    # Ouvre un fichier image
    chemin_image = filedialog.askopenfilename(
        title="Choisir une image",
        filetypes=[("Fichiers image", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )

    if chemin_image:
        try:
            image = Image.open(chemin_image)
            largeur, hauteur = image.size

            # Afficher la résolution
            label_resultat.config(
                text=f"Résolution : {largeur} x {hauteur} pixels"
            )

            # Afficher un aperçu réduit de l'image
            image.thumbnail((300, 300))
            image_tk = Image.PhotoImage(image)
            label_image.config(image=image_tk)
            label_image.image = image_tk  # Garde une référence

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image : {e}")

# Fenêtre principale
fenetre = tk.Tk()
fenetre.title("Lecteur de résolution d'image")
fenetre.geometry("400x450")

# Bouton
bouton_ouvrir = tk.Button(fenetre, text="Ajouter une image", command=ouvrir_image)
bouton_ouvrir.pack(pady=10)

# Label pour afficher l'image
label_image = tk.Label(fenetre)
label_image.pack()

# Label pour afficher la résolution
label_resultat = tk.Label(fenetre, text="Aucune image chargée", font=("Arial", 12))
label_resultat.pack(pady=10)

# Lancer l'application
fenetre.mainloop()
