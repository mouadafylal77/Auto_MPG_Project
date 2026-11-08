# Auto MPG : SVC et SVR, from scratch et scikit-learn

Projet Machine Learning e en Python : prédire trois classes de consommation à partir du MPG,
puis estimer directement le MPG. **Un MPG élevé signifie une consommation faible.**
Les trois modèles sont implémentés deux fois : SVC linéaire, SVC RBF et SVR linéaire
avec perte epsilon-insensible. Aucun estimateur SVM scikit-learn n'entraîne les modèles
from scratch ; SciPy résout leurs problèmes quadratiques explicites.

## Exécution rapide

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

`main.py` enchaîne automatiquement toutes les étapes : vérification des dépendances,
préparation des données, 19 tests pytest, exécution des deux notebooks et comparaison.
Il s'arrête immédiatement avec un message explicite si une étape échoue.

> **Première exécution :** le chargement du dataset nécessite Internet (seaborn met
> le CSV en cache dans `data/.cache/`). Les exécutions suivantes utilisent les fichiers locaux.

## Installation de l'environnement

Python CPython 3.11 est utilisé pour ce projet.

```powershell
Set-Location 'C:\Users\LENOVO\OneDrive\Bureau\Machine Learning\Auto_MPG_Project'
& 'C:\Users\LENOVO\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sur une autre machine, remplacer le chemin CPython par `py -3.11` ou un Python compatible.

## Architecture

```text
Auto_MPG_Project/
├── from_scratch/
│   ├── from_scratch.ipynb
│   ├── predictions/{svc_linear,svc_rbf,svr_linear}.csv
│   └── metrics/{classification_metrics,regression_metrics}.csv
├── scikit_learn/
│   ├── scikit_learn.ipynb
│   ├── predictions/{svc_linear,svc_rbf,svr_linear}.csv
│   └── metrics/{classification_metrics,regression_metrics}.csv
├── data/{train.csv,test.csv,class_thresholds.json}
├── comparison/
│   ├── {svc_linear,svc_rbf,svr_linear}_comparison.csv
│   ├── summary_metrics.csv
│   └── figures/
│       ├── svc_linear_comparison.png
│       ├── svc_rbf_comparison.png
│       ├── svr_real_vs_predicted.png
│       ├── epsilon_vs_mae.png
│       └── epsilon_vs_support_vectors.png
├── tests/{test_svc.py,test_svr.py}
├── prepare_data.py
├── compare.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Exécution étape par étape (débogage)

Si une étape échoue dans `main.py`, il est possible de la relancer seule :

```powershell
# Variables d'environnement (à définir dans chaque session PowerShell)
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:JUPYTER_PLATFORM_DIRS = '1'

# 1. Préparation des données
.\.venv\Scripts\python.exe prepare_data.py

# 2. Tests (19 tests attendus)
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 3. Notebook from scratch
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace `
    --ExecutePreprocessor.timeout=1800 from_scratch/from_scratch.ipynb

# 4. Notebook scikit-learn
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace `
    --ExecutePreprocessor.timeout=1800 scikit_learn/scikit_learn.ipynb

# 5. Comparaison
.\.venv\Scripts\python.exe compare.py
```

## Données et validation

- Source : `seaborn.load_dataset('mpg')`, 398 lignes, 6 valeurs de horsepower absentes.
  Suppression de ces lignes sans estimation : 392 observations.
- Partition externe : 80 % train (313 lignes), 20 % test (79), `random_state=16`.
- Les quantiles du train 1/3 et 2/3 définissent les classes 0 (consommation forte),
  1 (moyenne), 2 (faible). Les seuils sont enregistrés dans `class_thresholds.json`.
- Validation : 25 % du train, stratifiée, graine 16.
- StandardScaler sur les variables numériques ; One-Hot sur origin.
- Paramètres fixés : C SVC=1, gamma=1/p (p=9), C SVR=10.
  Epsilon choisi parmi 0.1, 0.5, 1, 2, 5 sur validation.

## Méthodes

Le SVC résout le dual à marge souple avec SLSQP et gradient analytique.
Noyaux linéaire et RBF ; trois problèmes binaires One-vs-One.

Le SVR résout le primal `0.5 ||w||² + C Σ(ξ)` avec slack par observation.
L'objectif est divisé par C avant optimisation pour stabiliser la recherche linéaire.

Tolérance SLSQP : 1e-9 ; maximum 3000 itérations. Une non-convergence déclenche une exception.

## Résultats mesurés

Exécution sous CPython **3.11.9**. Les **19 tests passent**.
Seuils train : **18.1 MPG** et **26.0 MPG**.

**SVC** (C=1 ; gamma RBF=1/9) — accord scratch/scikit-learn **100 %** :

| Modèle | F1 macro validation | Accuracy test | F1 macro test | Supports train |
|---|---:|---:|---:|---:|
| SVC linéaire | 0.763814 | 0.848101 | 0.854131 | 141 |
| SVC RBF | **0.813947** | 0.835443 | 0.837133 | 172 |

**SVR linéaire**, C=10 — from scratch (écarts scikit-learn infimes) :

| Epsilon | MAE validation | MAE test | RMSE test | R² test | Supports train |
|---|---:|---:|---:|---:|---:|
| **0.1** | **2.418785** | 2.731546 | 3.951376 | 0.784471 | 302 (96.49 %) |
| 0.5 | 2.429747 | 2.732572 | 3.992096 | 0.780006 | 269 (85.94 %) |
| 1 | 2.480177 | 2.729918 | 3.956279 | 0.783936 | 228 (72.84 %) |
| 2 | 2.513575 | 2.702506 | 3.819817 | 0.798584 | 148 (47.28 %) |
| 5 | 2.912655 | 2.868125 | 3.709776 | 0.810022 | 37 (11.82 %) |

**Epsilon=0.1 est retenu sur validation.** Accord scratch/scikit-learn : écart moyen < 0.00005 MPG.

## Limites

Ce petit jeu historique ne représente pas les véhicules modernes. Un holdout unique
ne quantifie pas l'incertitude. C et gamma ne font pas l'objet d'une recherche exhaustive.
Les QP denses sont adaptés à la pédagogie et à quelques centaines d'observations.
