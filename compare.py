"""Compare des configurations identiques par clé, jamais par ordre de ligne."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from prepare_data import ROOT, EPSILONS, class_labels, load_data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_checked(path, columns):
    require(Path(path).is_file(), f'Fichier absent : {path}')
    frame = pd.read_csv(path)
    require(set(columns) <= set(frame.columns), f'Colonnes nécessaires absentes : {path} : {set(columns)-set(frame.columns)}')
    require(not frame.empty and not frame.isna().any().any(), f'Données vides ou manquantes : {path}')
    numeric = frame.select_dtypes(include='number')
    require(np.isfinite(numeric.to_numpy()).all(), f'Valeurs non finies : {path}')
    return frame


def require_numeric(frame, columns, path):
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        except (ValueError, TypeError) as error:
            raise ValueError(f'Colonne numérique invalide {column} : {path}') from error
    require(np.isfinite(frame[columns].to_numpy(dtype=float)).all(), f'Valeurs numériques non finies : {path}')


def load_predictions(path, model, test, thresholds):
    regression = model == 'svr_linear'
    common = ['car_id', 'mpg_real', 'C', 'split']
    extra = ['mpg_predicted', 'signed_error', 'absolute_error', 'epsilon', 'outside_epsilon_tube'] if regression else ['classe_reelle', 'classe_predite', 'correct']
    if model == 'svc_rbf': extra += ['gamma']
    frame = read_checked(path, common+extra)
    require_numeric(frame, ['car_id', 'mpg_real', 'C']+(['epsilon', 'mpg_predicted', 'signed_error', 'absolute_error'] if regression else ['classe_reelle', 'classe_predite'])+(['gamma'] if model == 'svc_rbf' else []), path)
    keys = ['car_id', 'C']+(['epsilon'] if regression else ['gamma'] if model == 'svc_rbf' else [])
    require(not frame.duplicated(keys).any(), f'Clés dupliquées : {path}')
    require(set(frame.split) == {'test'}, f'Split différent du test : {path}')
    require((frame.C > 0).all(), f'C non positif : {path}')
    configurations = [k for k in keys if k != 'car_id']
    if regression:
        require(set(frame.epsilon) == set(EPSILONS), f'Epsilons attendus absents ou inattendus : {path}')
    else:
        require(len(frame[configurations].drop_duplicates()) == 1, f'Plusieurs configurations SVC : {path}')
        require(set(frame.classe_predite) <= {0, 1, 2}, f'Classes prédites invalides : {path}')
    if model == 'svc_rbf': require((frame.gamma > 0).all(), f'Gamma non positif : {path}')
    truth = test.set_index('car_id').mpg
    for _, part in frame.groupby(configurations, dropna=False):
        require(set(part.car_id) == set(test.car_id), f'Ensemble des identifiants test incomplet ou différent : {path}')
        values = truth.loc[part.car_id].to_numpy()
        require(np.allclose(values, part.mpg_real, rtol=0, atol=1e-10), f'Valeurs réelles MPG incompatibles : {path}')
        if regression:
            error = part.mpg_predicted.to_numpy()-values
            require(np.allclose(error, part.signed_error, atol=1e-9), f'Erreurs signées incohérentes : {path}')
            require(np.allclose(abs(error), part.absolute_error, atol=1e-9), f'Erreurs absolues incohérentes : {path}')
            require(np.array_equal(part.outside_epsilon_tube, abs(error)>part.epsilon.to_numpy()), f'Indicateur de tube incohérent : {path}')
        else:
            labels = class_labels(values, thresholds)
            require(np.array_equal(labels, part.classe_reelle), f'Classes réelles incompatibles : {path}')
            require(np.array_equal(part.correct, labels==part.classe_predite.to_numpy()), f'Indicateur correct incohérent : {path}')
    return frame


def align_predictions(first, second, keys):
    require(not first.duplicated(keys).any() and not second.duplicated(keys).any(), 'Clés dupliquées avant jointure.')
    joined = first.merge(second, on=keys, how='outer', suffixes=('_scratch', '_sklearn'),
                         indicator=True, validate='one_to_one')
    require((joined._merge == 'both').all(), 'Identifiants ou configurations incompatibles entre implémentations.')
    require(np.allclose(joined.mpg_real_scratch, joined.mpg_real_sklearn, rtol=0, atol=1e-10), 'Valeurs réelles différentes entre implémentations.')
    return joined.drop(columns='_merge')


def read_metrics(implementation, regression, train_size):
    filename = 'regression_metrics.csv' if regression else 'classification_metrics.csv'
    required = ['model', 'C', 'split', 'n_support'] + (['epsilon', 'mae', 'mse', 'rmse', 'r2', 'support_percentage', 'validation_mae'] if regression else ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'confusion_matrix', 'validation_accuracy', 'validation_f1_macro', 'gamma'])
    metrics = read_checked(ROOT / implementation / 'metrics' / filename, required)
    require_numeric(metrics, [c for c in required if c not in ['model','split','confusion_matrix']], filename)
    keys = ['model', 'C']+(['epsilon'] if regression else [])
    require(not metrics.duplicated(keys).any(), f'Métriques dupliquées : {implementation}')
    require(set(metrics.split) == {'test'}, f'Split métriques invalide : {implementation}')
    require(((metrics.n_support >= 0) & (metrics.n_support <= train_size)).all(), f'Nombre de supports invalide : {implementation}')
    require((metrics.n_support == metrics.n_support.astype(int)).all(), f'Nombre de supports non entier : {implementation}')
    if regression:
        require(set(metrics.model) == {'svr_linear'} and set(metrics.epsilon) == set(EPSILONS), f'Configurations métriques SVR invalides : {implementation}')
        require(np.allclose(metrics.support_percentage, 100*metrics.n_support/train_size), f'Pourcentage supports incohérent : {implementation}')
    else:
        require(set(metrics.model) == {'svc_linear', 'svc_rbf'} and len(metrics) == 2, f'Modèles métriques SVC invalides : {implementation}')
    return metrics


def regression_scores(real, pred):
    return dict(mae=mean_absolute_error(real, pred), mse=mean_squared_error(real, pred),
                rmse=np.sqrt(mean_squared_error(real, pred)), r2=r2_score(real, pred))


def main():
    train, test, thresholds = load_data()
    implementations = ['from_scratch', 'scikit_learn']
    models = ['svc_linear', 'svc_rbf', 'svr_linear']
    # Tout valider avant de produire les comparaisons.
    predictions = {(impl, model): load_predictions(ROOT / impl / 'predictions' / f'{model}.csv', model, test, thresholds)
                   for impl in implementations for model in models}
    metrics = {(impl, reg): read_metrics(impl, reg, len(train)) for impl in implementations for reg in [False, True]}
    aligned = {model: align_predictions(predictions[('from_scratch', model)], predictions[('scikit_learn', model)],
                    ['car_id', 'C']+(['epsilon'] if model == 'svr_linear' else ['gamma'] if model == 'svc_rbf' else [])) for model in models}
    output, figures = ROOT / 'comparison', ROOT / 'comparison/figures'
    figures.mkdir(parents=True, exist_ok=True)
    summary, reports = [], {}
    for model in models[:2]:
        joined = aligned[model]
        joined['agreement'] = joined.classe_predite_scratch == joined.classe_predite_sklearn
        joined['agreement_percentage'] = 100*joined.agreement.mean()
        joined['n_different'] = (~joined.agreement).sum()
        scores = {}
        for impl, suffix in zip(implementations, ['scratch', 'sklearn']):
            m = metrics[(impl, False)].set_index('model').loc[model]
            require(np.isclose(m.C, joined.C.iloc[0]), f'C métriques incohérent : {impl}/{model}')
            if model == 'svc_rbf': require(np.isclose(m.gamma, joined.gamma.iloc[0]), f'Gamma métriques incohérent : {impl}/{model}')
            scores[suffix] = dict(accuracy=accuracy_score(joined.classe_reelle_scratch, joined[f'classe_predite_{suffix}']),
                f1_macro=f1_score(joined.classe_reelle_scratch, joined[f'classe_predite_{suffix}'], average='macro'))
            from sklearn.metrics import confusion_matrix, precision_score, recall_score
            require(np.array_equal(json.loads(m.confusion_matrix), confusion_matrix(joined.classe_reelle_scratch, joined[f'classe_predite_{suffix}'], labels=[0,1,2])), f'Matrice de confusion incohérente : {impl}/{model}')
            require(np.isclose(m.precision_macro, precision_score(joined.classe_reelle_scratch, joined[f'classe_predite_{suffix}'], average='macro', zero_division=0)), f'Précision incohérente : {impl}/{model}')
            require(np.isclose(m.recall_macro, recall_score(joined.classe_reelle_scratch, joined[f'classe_predite_{suffix}'], average='macro', zero_division=0)), f'Rappel incohérent : {impl}/{model}')
            best_model = metrics[(impl, False)].sort_values(['validation_f1_macro', 'model'], ascending=[False,True]).iloc[0].model
            for metric, value in scores[suffix].items():
                require(np.isclose(value, m[metric]), f'Métrique {metric} incohérente : {impl}/{model}')
                joined[f'{metric}_{suffix}'] = value
                summary.append(dict(implementation=impl, task='classification', model=model, C=m.C, gamma=m.gamma,
                    epsilon=0., selected_on_validation=model==best_model, validation_metric='f1_macro',
                    validation_value=m.validation_f1_macro, test_metric=metric, test_value=value, n_support=int(m.n_support)))
        for metric in ['accuracy', 'f1_macro']:
            joined[f'{metric}_delta_scratch_minus_sklearn'] = scores['scratch'][metric]-scores['sklearn'][metric]
        reports[model] = joined
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(2)
        for offset, suffix in [(-.18, 'scratch'), (.18, 'sklearn')]:
            axes[0].bar(x+offset, [scores[suffix]['accuracy'], scores[suffix]['f1_macro']], width=.36, label=suffix)
        axes[0].set(xticks=x, xticklabels=['Accuracy', 'F1 macro'], ylim=(0,1), title=f'{model} — test', ylabel='Score')
        axes[0].legend()
        axes[1].bar(['Accord', 'Désaccord'], [joined.agreement.sum(), (~joined.agreement).sum()])
        axes[1].set(title=f'Accord : {100*joined.agreement.mean():.2f} %', ylabel='Nombre de voitures')
        fig.tight_layout(); fig.savefig(figures / f'{model}_comparison.png', dpi=160); plt.close(fig)

    joined = aligned['svr_linear']
    joined['prediction_delta_scratch_minus_sklearn'] = joined.mpg_predicted_scratch-joined.mpg_predicted_sklearn
    joined['absolute_prediction_delta'] = abs(joined.prediction_delta_scratch_minus_sklearn)
    regression_parts = []
    for (C, epsilon), part in joined.groupby(['C', 'epsilon']):
        part = part.copy()
        for impl, suffix in zip(implementations, ['scratch', 'sklearn']):
            table = metrics[(impl, True)]
            m = table[(table.C == C) & (table.epsilon == epsilon)]
            require(len(m) == 1, f'Configuration métriques SVR absente : {impl}, C={C}, epsilon={epsilon}')
            m = m.iloc[0]
            chosen = table.sort_values(['validation_mae', 'epsilon']).iloc[0]
            scores = regression_scores(part.mpg_real_scratch, part[f'mpg_predicted_{suffix}'])
            for metric, value in scores.items():
                require(np.isclose(value, m[metric]), f'Métrique SVR {metric} incohérente : {impl}/{epsilon}')
                part[f'{metric}_{suffix}'] = value
                summary.append(dict(implementation=impl, task='regression', model='svr_linear', C=C, gamma=0., epsilon=epsilon,
                    selected_on_validation=(epsilon==chosen.epsilon and C==chosen.C), validation_metric='mae',
                    validation_value=m.validation_mae, test_metric=metric, test_value=value, n_support=int(m.n_support)))
            part[f'n_support_{suffix}'] = int(m.n_support)
        for metric in ['mae','rmse','r2','n_support']:
            part[f'{metric}_delta_scratch_minus_sklearn'] = part[f'{metric}_scratch']-part[f'{metric}_sklearn']
        part['mean_absolute_prediction_delta'] = part.absolute_prediction_delta.mean()
        part['max_absolute_prediction_delta'] = part.absolute_prediction_delta.max()
        regression_parts.append(part)
    reports['svr_linear'] = pd.concat(regression_parts, ignore_index=True)
    summary = pd.DataFrame(summary)
    for model, report in reports.items():
        require(not report.isna().any().any(), f'Résultats comparatifs manquants : {model}')
        report.to_csv(output / f'{model}_comparison.csv', index=False)
    summary.to_csv(output / 'summary_metrics.csv', index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    for impl in implementations:
        table = metrics[(impl, True)]
        chosen = table.sort_values(['validation_mae', 'epsilon']).iloc[0]
        frame = predictions[(impl, 'svr_linear')]
        frame = frame[(frame.epsilon==chosen.epsilon) & (frame.C==chosen.C)]
        ax.scatter(frame.mpg_real, frame.mpg_predicted, s=25, alpha=.6, label=f'{impl}, epsilon={chosen.epsilon}')
    limits = [test.mpg.min()-2, test.mpg.max()+2]
    ax.plot(limits, limits, 'k--', label='Prédiction parfaite')
    ax.set(title='SVR choisi sur validation — test', xlabel='MPG réel', ylabel='MPG prédit'); ax.legend()
    fig.tight_layout(); fig.savefig(figures / 'svr_real_vs_predicted.png', dpi=160); plt.close(fig)
    for column, filename, ylabel in [('mae','epsilon_vs_mae.png','MAE test (MPG)'), ('n_support','epsilon_vs_support_vectors.png','Nombre de supports train')]:
        fig, ax = plt.subplots(figsize=(7,4))
        for impl in implementations:
            table = metrics[(impl, True)].sort_values('epsilon')
            ax.plot(table.epsilon, table[column], 'o-', label=impl)
        ax.set(title=ylabel+' en fonction d’epsilon', xlabel='Epsilon (MPG)', ylabel=ylabel); ax.legend()
        fig.tight_layout(); fig.savefig(figures / filename, dpi=160); plt.close(fig)
    print('\nTableau comparatif (choix fondés uniquement sur validation) :')
    print(summary.to_string(index=False))
    print('\nConfigurations retenues :')
    print(summary[summary.selected_on_validation & summary.test_metric.isin(['accuracy','mae'])].to_string(index=False))
    for model in models[:2]:
        report = reports[model]
        print(f'{model} : accord={report.agreement_percentage.iloc[0]:.2f} %, différences={report.n_different.iloc[0]}')
    print('SVR : écarts de prédiction par configuration :')
    print(reports['svr_linear'].groupby(['C','epsilon'])[['mean_absolute_prediction_delta','max_absolute_prediction_delta','n_support_delta_scratch_minus_sklearn']].first().to_string())


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f'Comparaison arrêtée : {error}') from error
