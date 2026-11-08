# Auto MPG : SVC et SVR, from scratch et scikit-learn

Projet universitaire en Python : prédire trois classes de consommation à partir du MPG,
puis estimer directement le MPG. **Un MPG élevé signifie une consommation faible.**
Les trois modèles sont implémentés deux fois : SVC linéaire, SVC RBF et SVR linéaire
avec perte epsilon-insensible. Aucun estimateur SVM scikit-learn n'entraîne les modèles
from scratch ; SciPy résout leurs problèmes quadratiques explicites.

## Architecture et livrables

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
├── requirements.txt
├── README.md
└── .gitignore
```

Les deux notebooks contiennent les explications, le code, les tableaux et les graphiques.
Les implémentations from scratch sont visibles dans les cellules, et non dans un module
externe. Les tests extraient ces cellules étiquetées pour vérifier exactement le code livré.

## Installation Windows PowerShell

Python CPython 3.11 est utilisé pour cette exécution. Le Python MSYS2 trouvé par défaut
sur cette machine ne possédait pas pip : utiliser CPython pour créer l'environnement.
NumPy, SciPy >=1.16 (multiplicateurs SLSQP), pandas, scikit-learn, seaborn, matplotlib,
Jupyter/nbconvert, ipykernel et pytest sont nécessaires.

```powershell
Set-Location 'C:\Users\LENOVO\OneDrive\Bureau\Machine Learning\Auto_MPG_Project'
& 'C:\Users\LENOVO\AppData\Local\Programs\Python\Python311\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sur une autre machine, remplacer le chemin CPython par `py -3.11` ou un Python compatible.
L'activation PowerShell est facultative : les commandes suivantes utilisent directement
l'interpréteur local. Le premier chargement du dataset nécessite Internet ; seaborn met
sa source en cache dans `data/.cache/` (ignoré par Git). Une fois les CSV créés,
notebooks, tests et comparaisons utilisent les fichiers locaux.

## Reproduire les expériences

Depuis la racine du projet, exécuter dans cet ordre :

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:JUPYTER_PLATFORM_DIRS = '1'
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook from_scratch/from_scratch.ipynb --output from_scratch.ipynb --ExecutePreprocessor.timeout=1800
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook scikit_learn/scikit_learn.ipynb --output scikit_learn.ipynb --ExecutePreprocessor.timeout=1800
.\.venv\Scripts\python.exe compare.py
```

`nbconvert` écrit les sorties dans le notebook source situé dans chaque dossier.
Les chemins sont résolus à partir de la racine ; on peut aussi ouvrir chaque notebook
depuis son dossier. Le kernel doit utiliser `.venv` ; lancé via `python -m nbconvert`,
le kernel Python générique hérite de l'interpréteur de cet environnement.
Si nécessaire, dans l'interface Jupyter choisir le Python de `.venv`.

## Données et validation

- Source : `seaborn.load_dataset('mpg')`, 398 lignes, 6 valeurs de horsepower absentes.
  Suppression de ces lignes sans estimation : 392 observations. `car_id` est l'index
  original, attribué avant la suppression. `name` est conservé pour l'inspection mais
  exclu des modèles, car quasi unique et peu généralisable.
- Partition externe : 80 % train (313 lignes), 20 % test (79), `random_state=16`.
  Pas de stratification fondée sur des seuils appris avant cette partition.
- Les quantiles du train 1/3 et 2/3 définissent les classes 0 (consommation forte),
  1 (moyenne), 2 (faible). `mpg <= q1` donne 0 ; `q1 < mpg <= q2` donne 1 ; sinon 2.
  Les seuils et leur définition sont enregistrés dans `class_thresholds.json`.
- Validation : 25 % du train, stratifiée selon cette cible commune, graine 16.
  Les seuils constituent une définition de la cible sur le train externe ; ils restent
  fixes pendant la validation. Les caractéristiques de validation ne participent
  jamais à l'ajustement du scaler ou de l'encodeur du sous-train.
- StandardScaler sur cylinders, displacement, horsepower, weight, acceleration,
  model_year ; One-Hot sur origin. Cylinders est un compte ordonné numérique ; ce
  choix suppose un effet linéaire pour les modèles linéaires et reste discutable.
  Les indicatrices ne sont pas standardisées. Aucun ajustement n'utilise le test.
- Paramètres fixés a priori pour comparer les solveurs : C SVC=1, gamma=1/p (p=9),
  C SVR=10. Chaque notebook mesure sa propre validation avec les mêmes configurations.
  Choix du noyau par F1 macro de validation ; choix d'epsilon par MAE de validation
  parmi 0.1, 0.5, 1, 2, 5. Les égalités sont départagées par nom de modèle / epsilon.
  Les cinq epsilons sont évalués sur le test pour l'étude de sensibilité prédéfinie ;
  ils ne servent pas à sélectionner après observation du test.

## Méthodes et contrôles mathématiques

Le SVC résout le dual à marge souple sous les contraintes `0 <= alpha <= C` et
`alpha @ y = 0` avec SLSQP et gradient analytique. Noyau linéaire ou RBF ; trois
problèmes binaires One-vs-One. Le biais est calculé sur les supports libres, sinon
à partir des bornes KKT. La liste globale de supports est une union sans doublon
des supports des paires. Le vote choisit la première classe ordonnée en cas d'égalité.

Le SVR résout le primal `0.5 ||w||² + C sum(xi)` avec un slack par observation et
deux contraintes par point : `y-f <= epsilon+xi` et `f-y <= epsilon+xi`.
Cette formulation équivaut à la perte epsilon-insensible. Les multiplicateurs des
contraintes donnent alpha et alpha*, donc beta=alpha-alpha*. Le biais n'est pas
régularisé et MPG reste dans son unité d'origine. Tout l'objectif primal est divisé
par C avant optimisation pour stabiliser la recherche linéaire, sans modifier
l'optimum ; les multiplicateurs du solveur sont ensuite multipliés par C.

Tolérance objectif SLSQP : 1e-9 ; maximum 3000 itérations ; seuil alpha SVC : 1e-6 ;
seuil alpha+alpha* SVR : 1e-5. La référence libsvm utilise `tol=1e-7`, sans shrinking.
Une non-convergence déclenche une exception. Les notebooks affichent les diagnostics
KKT SVC, faisabilité, stationnarité et écart primal-dual SVR ; les tests vérifient
aussi ces propriétés sur données synthétiques, les noyaux, les supports et les votes.
La classification n'est pas remplacée par une régression, ni le SVR par moindres carrés.

## Format des CSV et métriques

Les quatre CSV SVC contiennent `car_id`, `mpg_real`, `classe_reelle`, `classe_predite`,
`correct`, `C`, `split` et `gamma` pour RBF. Une ligne par voiture test.
Les deux CSV SVR contiennent `car_id`, `mpg_real`, `mpg_predicted`, `signed_error`
(prédit moins réel), `absolute_error`, `epsilon`, `C`, `outside_epsilon_tube`, `split`.
Une ligne par voiture et epsilon : 395 lignes par implémentation. L'indicateur de
tube sur le test ne définit pas les supports, qui sont des points d'entraînement.

Métriques SVC : accuracy, précision/rappel/F1 macro, matrice de confusion sérialisée
en JSON, supports uniques et scores de validation. Métriques SVR : MAE, MSE, RMSE,
R², nombre et pourcentage de supports train, epsilon/C, MAE de validation.
Les métriques sont recalculées à partir des vraies prédictions.

`compare.py` vérifie l'existence et les colonnes des fichiers, les clés uniques,
les identifiants et cibles du test de référence, les configurations communes, les
erreurs exportées et la cohérence des métriques. Les jointures sont réalisées par
`car_id`, C, gamma/epsilon. Un fichier incompatible interrompt la comparaison avec
un message explicite. Les comparaisons détaillées conservent les prédictions par
voiture, les écarts et les statistiques par configuration. Le nombre de supports
provient des métriques d'entraînement, pas du nombre d'erreurs test.

`summary_metrics.csv` est au format long : une ligne par implémentation, modèle,
configuration et métrique test. `selected_on_validation` indique les choix ;
`validation_metric` et `validation_value` documentent leur justification.
Gamma=0 pour un modèle linéaire et epsilon=0 pour un SVC signifient « non applicable ».
Les graphiques comparatifs sont régénérés automatiquement par `compare.py`.

## Résultats mesurés

Exécution réelle sous CPython **3.11.9**, avec les versions fixées dans
`requirements.txt`. Les deux notebooks ont été exécutés intégralement avec
`nbconvert`, leurs tableaux et figures sont enregistrés dans les `.ipynb`.
Les **19 tests passent** ; `pip check` ne signale aucun conflit.

Seuils train mesurés : **18.1 MPG** et **26.0 MPG**. Résultats SVC identiques
pour les deux implémentations (C=1 ; gamma RBF=1/9) :

| Modèle | F1 macro validation | Accuracy test | F1 macro test | Supports train |
|---|---:|---:|---:|---:|
| SVC linéaire | 0.763814 | 0.848101 | 0.854131 | 141 |
| SVC RBF | **0.813947** | 0.835443 | 0.837133 | 172 |

Le **RBF est retenu par validation**, avec 83.54 % d'accuracy test. Le linéaire
obtient 84.81 % sur ce test ; cette observation ne modifie pas le choix préalable.
L'accord scratch/scikit-learn est **100 %** pour chacun des SVC, soit 0 voiture
classée différemment. Les matrices de confusion test, lignes=réel / colonnes=prédit,
ordre des classes 0, 1, 2, sont :

```text
Linéaire : [[18, 3, 0], [0, 21, 3], [0, 6, 28]]
RBF      : [[19, 2, 0], [2, 19, 3], [0, 6, 28]]
```

SVR linéaire, C=10, résultats from scratch (les écarts scikit-learn sont infimes) :

| Epsilon | MAE validation | MAE test | RMSE test | R² test | Supports train |
|---|---:|---:|---:|---:|---:|
| **0.1** | **2.418785** | 2.731546 | 3.951376 | 0.784471 | 302 (96.49 %) |
| 0.5 | 2.429747 | 2.732572 | 3.992096 | 0.780006 | 269 (85.94 %) |
| 1 | 2.480177 | 2.729918 | 3.956279 | 0.783936 | 228 (72.84 %) |
| 2 | 2.513575 | 2.702506 | 3.819817 | 0.798584 | 148 (47.28 %) |
| 5 | 2.912655 | 2.868125 | 3.709776 | 0.810022 | 37 (11.82 %) |

**Epsilon=0.1 est choisi sur validation dans les deux implémentations.** Sur le
test, scikit-learn obtient MAE=2.731546, RMSE=3.951371 et R²=0.784472 pour cette
configuration. Epsilon=2 présente une MAE test plus faible, mais n'est pas retenu
après examen du test. Dans cette expérience, élargir le tube réduit fortement le
nombre de supports ; les effets sur MAE et R² ne sont pas identiques.

| Epsilon | Écart absolu moyen des prédictions (MPG) | Écart absolu maximal (MPG) |
|---|---:|---:|
| 0.1 | 0.000014 | 0.000063 |
| 0.5 | 0.000027 | 0.000075 |
| 1 | 0.000049 | 0.000153 |
| 2 | 0.000011 | 0.000043 |
| 5 | 0.000016 | 0.000053 |

Les nombres de supports sont identiques pour les cinq epsilons. Ces résultats
étayent l'équivalence des solutions obtenues sur cette expérience, complétée par
les vérifications mathématiques ; ils ne prouvent pas une correction universelle.

Artefacts vérifiés : 6 CSV de prédictions (79 lignes par SVC, 395 par SVR),
4 CSV de métriques, 3 CSV de comparaison détaillée et 1 synthèse (48 lignes),
5 figures comparatives et 2 notebooks exécutés (6 graphiques enregistrés chacun).
Aucune valeur manquante ou non finie n'a été détectée dans les résultats exportés.
Les figures ont aussi été inspectées visuellement. Une régénération des partitions
et des seuils depuis le cache a produit exactement les mêmes empreintes SHA-256.
Les diagnostics finaux donnent un résidu KKT SVC maximal d'environ 3.4e-5 et un
écart primal-dual maximal d'environ 7.6e-5. Pour les SVR, l'écart primal-dual absolu
reste inférieur à 1e-7 ; ses très petites valeurs négatives viennent de l'arrondi
numérique, et non d'une inversion du sens de la dualité faible.

### Difficultés résolues

- Le Python MSYS2 initial n'avait ni pip ni dépendances : environnement `.venv`
  créé avec CPython 3.11.9 et dépendances installées et vérifiées.
- Le réseau restreint bloquait les téléchargements : installation des paquets et
  premier chargement seaborn effectués avec l'accès nécessaire, puis usage local.
- Jupyter était bloqué avant exécution par les connexions locales interdites dans
  le bac à sable. Exécution avec accès aux sockets locales pour lancer le noyau.
  Cela ne nécessite aucune modification des algorithmes sur un Windows ordinaire.
- Le SVR primal non normalisé déclenchait `Positive directional derivative for
  linesearch` sur Auto MPG. Division de tout l'objectif par C, remise à l'échelle
  des multiplicateurs et ajout d'un test de faisabilité/dualité sur le sous-train
  réel. Les 19 tests et le notebook complet passent après cette correction.

## Limites

Ce petit jeu historique ne représente pas les véhicules modernes. Un holdout unique
ne quantifie pas l'incertitude des performances. Les seuils transforment une cible
continue en classes arbitraires ; MPG n'est pas une mesure en litres/100 km.
C et gamma ne font pas l'objet d'une recherche exhaustive. Une validation croisée
imbriquée constituerait une extension. Les QP denses ont un coût élevé en temps et
mémoire ; ils sont ici adaptés à la pédagogie et à quelques centaines d'observations.
Les supports proches de zéro peuvent varier avec les tolérances. Quand le tube SVR
contient tous les points, le biais peut être non unique : l'objectif compte alors
davantage qu'une égalité parfaite des prédictions entre solveurs.

Les environnements, caches et fichiers temporaires sont ignorés par Git ; les
notebooks exécutés, CSV et figures d'évaluation restent disponibles. Aucun commit
ni push n'est effectué par les scripts.
