import subprocess

# Liste des scripts à exécuter dans l'ordre
scripts = [
    "scripts/1_generation.py",
    "scripts/1.1Data_visualization_stage.py",
    "scripts/2_Synthetic_noisy_data_generation.py",
    "scripts/3_backprojection.py",
    "scripts/algo_iteratif4.py",
    "scripts/5_save.py",
    "scripts/6_entrainement.py",  # Si tu n'entraînes pas de modèle, tu peux commenter cette ligne
    "scripts/7_comparer.py"
]

for i, script in enumerate(scripts, 1):
    print(f"\n--- Étape {i}: Exécution de {script} ---\n")
    try:
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de {script} : {e}")
        break
