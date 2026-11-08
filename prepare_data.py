"""Partitions et protocole communs, sans ajustement sur le test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 16
NUMERIC = ['cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model_year']
FEATURES = NUMERIC + ['origin']
EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0]
SVC_C, SVR_C = 1.0, 10.0
CLASS_NAMES = ['Consommation forte', 'Consommation moyenne', 'Consommation faible']


def class_labels(mpg, thresholds):
    """0: MPG bas, donc consommation forte ; 2: MPG haut, consommation faible."""
    return np.searchsorted([thresholds['q1'], thresholds['q2']], mpg, side='left')


def make_preprocessor():
    # Cylinders est une quantité ordonnée, conservée comme variable numérique.
    return ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC),
        ('origin', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['origin'])
    ], remainder='drop', sparse_threshold=0)


def prepare():
    raw = sns.load_dataset('mpg', data_home=ROOT / 'data/.cache')
    print('Dimensions brutes :', raw.shape)
    print('Valeurs manquantes :\n', raw.isna().sum().to_string())
    if len(raw) != 398 or raw.index.has_duplicates:
        raise ValueError('Source MPG inattendue : 398 identifiants uniques attendus.')
    raw = raw.copy()
    raw.insert(0, 'car_id', raw.index.astype(int))
    # Six valeurs de horsepower absentes : suppression avant partition, sans estimation.
    clean = raw.dropna(subset=FEATURES + ['mpg']).copy()
    train, test = train_test_split(clean, test_size=0.2, random_state=SEED)
    q1, q2 = train.mpg.quantile([1/3, 2/3]).to_numpy()
    thresholds = dict(q1=float(q1), q2=float(q2), random_state=SEED,
                      test_size=0.2, raw_rows=len(raw), complete_rows=len(clean),
                      labels={str(i): name for i, name in enumerate(CLASS_NAMES)},
                      source='seaborn.load_dataset("mpg")',
                      definition='classe 0: mpg <= q1; classe 1: q1 < mpg <= q2; classe 2: mpg > q2')
    (ROOT / 'data').mkdir(exist_ok=True)
    for name, frame in [('train', train), ('test', test)]:
        frame.to_csv(ROOT / 'data' / f'{name}.csv', index=False)
    (ROOT / 'data/class_thresholds.json').write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Observations complètes : {len(clean)} ; train : {len(train)} ; test : {len(test)}')
    print('Seuils calculés uniquement sur le train :', q1, q2)


def load_data():
    train = pd.read_csv(ROOT / 'data/train.csv')
    test = pd.read_csv(ROOT / 'data/test.csv')
    thresholds = json.loads((ROOT / 'data/class_thresholds.json').read_text(encoding='utf-8'))
    if set(train.car_id) & set(test.car_id):
        raise ValueError('Partitions non disjointes.')
    return train, test, thresholds


def validation_split(train, thresholds):
    """Holdout 25 % du train ; aucune transformation n'est encore ajustée."""
    return train_test_split(train, test_size=0.25, random_state=SEED,
                            stratify=class_labels(train.mpg, thresholds))


if __name__ == '__main__':
    prepare()
