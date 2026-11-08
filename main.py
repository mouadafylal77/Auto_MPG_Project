"""Point d'entrée unique du projet Auto_MPG_Project.

Usage :
    .\.venv\Scripts\python.exe main.py
"""
import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

REQUIRED_PACKAGES = [
    "numpy", "scipy", "pandas", "sklearn", "seaborn",
    "matplotlib", "nbformat", "nbconvert", "ipykernel", "pytest",
]

NOTEBOOKS = [
    ROOT / "from_scratch" / "from_scratch.ipynb",
    ROOT / "scikit_learn" / "scikit_learn.ipynb",
]

EXPECTED_FILES = [
    ROOT / "from_scratch" / "predictions" / "svc_linear.csv",
    ROOT / "from_scratch" / "predictions" / "svc_rbf.csv",
    ROOT / "from_scratch" / "predictions" / "svr_linear.csv",
    ROOT / "from_scratch" / "metrics" / "classification_metrics.csv",
    ROOT / "from_scratch" / "metrics" / "regression_metrics.csv",
    ROOT / "scikit_learn" / "predictions" / "svc_linear.csv",
    ROOT / "scikit_learn" / "predictions" / "svc_rbf.csv",
    ROOT / "scikit_learn" / "predictions" / "svr_linear.csv",
    ROOT / "scikit_learn" / "metrics" / "classification_metrics.csv",
    ROOT / "scikit_learn" / "metrics" / "regression_metrics.csv",
    ROOT / "comparison" / "svc_linear_comparison.csv",
    ROOT / "comparison" / "svc_rbf_comparison.csv",
    ROOT / "comparison" / "svr_linear_comparison.csv",
    ROOT / "comparison" / "summary_metrics.csv",
    ROOT / "comparison" / "figures" / "svc_linear_comparison.png",
    ROOT / "comparison" / "figures" / "svc_rbf_comparison.png",
    ROOT / "comparison" / "figures" / "svr_real_vs_predicted.png",
    ROOT / "comparison" / "figures" / "epsilon_vs_mae.png",
    ROOT / "comparison" / "figures" / "epsilon_vs_support_vectors.png",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(text: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def step(text: str) -> None:
    print(f"\n>>> {text}")


def abort(message: str) -> None:
    print(f"\n[ERREUR] {message}", file=sys.stderr)
    sys.exit(1)


def run(args: list, label: str, cwd: Path | None = None) -> None:
    """Lance une commande et arrête le programme en cas d'échec."""
    result = subprocess.run(
        args,
        cwd=str(cwd or ROOT),
        env=os.environ,
    )
    if result.returncode != 0:
        abort(f"Étape échouée : {label} (code {result.returncode})")


# ---------------------------------------------------------------------------
# Étapes
# ---------------------------------------------------------------------------
def check_dependencies() -> None:
    banner("1/6 — Vérification des dépendances")
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        abort(
            f"Bibliothèque(s) manquante(s) : {', '.join(missing)}\n"
            f"Installez-les avec :\n"
            f"  .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )
    print("Toutes les dépendances sont présentes.")


def configure_env() -> None:
    banner("2/6 — Configuration des variables d'environnement")
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "JUPYTER_PLATFORM_DIRS": "1",
    }
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")


def run_prepare_data() -> None:
    banner("3/6 — Préparation des données (prepare_data.py)")
    run([PYTHON, str(ROOT / "prepare_data.py")], "prepare_data.py")
    print("Données préparées avec succès.")


def run_tests() -> None:
    banner("4/6 — Tests pytest (19 tests attendus)")
    run(
        [PYTHON, "-m", "pytest", str(ROOT / "tests"), "-v", "--tb=short"],
        "pytest",
    )
    print("Tous les tests ont réussi.")


def run_notebooks() -> None:
    banner("5/6 — Exécution des notebooks avec nbconvert")
    for notebook in NOTEBOOKS:
        step(f"Exécution : {notebook.relative_to(ROOT)}")
        run(
            [
                PYTHON, "-m", "nbconvert",
                "--execute",
                "--to", "notebook",
                "--inplace",
                f"--ExecutePreprocessor.timeout=1800",
                str(notebook),
            ],
            f"nbconvert {notebook.name}",
            cwd=notebook.parent,
        )
        print(f"  {notebook.name} exécuté et sauvegardé.")


def run_compare() -> None:
    banner("6/6 — Comparaison des implémentations (compare.py)")
    run([PYTHON, str(ROOT / "compare.py")], "compare.py")
    print("Comparaison terminée avec succès.")


def print_summary() -> None:
    banner("Résumé final")

    # Métriques depuis summary_metrics.csv
    summary_path = ROOT / "comparison" / "summary_metrics.csv"
    if summary_path.is_file():
        try:
            import pandas as pd
            summary = pd.read_csv(summary_path)
            selected = summary[
                summary["selected_on_validation"]
                & summary["test_metric"].isin(["accuracy", "mae"])
            ][["implementation", "task", "model", "epsilon", "test_metric", "test_value", "n_support"]]
            print("\nConfigurations retenues sur validation :")
            print(selected.to_string(index=False))
        except Exception as exc:
            print(f"  (Impossible de lire summary_metrics.csv : {exc})")
    else:
        print("  summary_metrics.csv introuvable.")

    # Fichiers générés
    print("\nFichiers générés :")
    missing = []
    for path in EXPECTED_FILES:
        rel = path.relative_to(ROOT)
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"  [OK] {rel}  ({size_kb:.1f} Ko)")
        else:
            print(f"  [MANQUANT] {rel}")
            missing.append(rel)

    if missing:
        print(f"\nAttention : {len(missing)} fichier(s) attendu(s) manquant(s).")
    else:
        print(f"\nTous les {len(EXPECTED_FILES)} fichiers attendus sont présents.")

    print("\n[SUCCÈS] Pipeline complet terminé.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    banner("Auto MPG Project — Pipeline complet")
    print(f"Python : {PYTHON}")
    print(f"Racine : {ROOT}")

    check_dependencies()
    configure_env()
    run_prepare_data()
    run_tests()
    run_notebooks()
    run_compare()
    print_summary()


if __name__ == "__main__":
    main()
