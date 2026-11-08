"""Les tests exécutent directement les définitions du notebook livré."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import nbformat
import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def definitions():
    namespace = {}
    notebook = nbformat.read(ROOT / 'from_scratch/from_scratch.ipynb', as_version=4)
    for cell in notebook.cells:
        if 'scratch_svc' in cell.metadata.get('tags', []):
            exec(compile(cell.source, 'from_scratch.ipynb:svc', 'exec'), namespace)
    return namespace


def test_rbf_kernel():
    kernel = definitions()['rbf_kernel']
    X = np.array([[0., 0.], [1., 0.], [1., 2.]])
    K = kernel(X, X, .5)
    np.testing.assert_allclose(np.diag(K), 1)
    np.testing.assert_allclose(K, K.T)
    assert K[0, 1] == pytest.approx(np.exp(-.5))
    assert np.linalg.eigvalsh(K).min() >= -1e-10


@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_binary_dual_kkt_and_reference(kernel):
    X, y = make_blobs(n_samples=40, centers=2, random_state=16, cluster_std=1.8)
    X = StandardScaler().fit_transform(X)
    y = 2*y-1
    model = definitions()['BinarySVC'](C=1., kernel=kernel, gamma=.7).fit(X, y)
    reference = SVC(C=1., kernel=kernel, gamma=.7, tol=1e-9, shrinking=False).fit(X, y)
    assert model.result_.success
    assert model.equality_residual_ < 1e-7
    assert model.alpha_.min() >= -1e-8 and model.alpha_.max() <= 1+1e-8
    assert model.kkt_violation_ < 2e-4
    assert abs(model.duality_gap_) < 2e-4
    assert len(model.support_) > 0
    assert np.array_equal(model.support_, np.flatnonzero(model.alpha_ > 1e-6))
    np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X), atol=3e-4)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    if kernel == 'linear':
        np.testing.assert_allclose(model.decision_function(X), X@model.coef_+model.intercept_)


@pytest.mark.parametrize('kernel', ['linear', 'rbf'])
def test_three_classes_ovo(kernel):
    X, y = make_blobs(n_samples=45, centers=3, cluster_std=.7, random_state=16)
    X = StandardScaler().fit_transform(X)
    model = definitions()['ScratchSVC'](C=2, kernel=kernel, gamma=.5).fit(X, y)
    reference = SVC(C=2, kernel=kernel, gamma=.5, tol=1e-9).fit(X, y)
    assert len(model.models_) == 3
    assert model.decision_function(X).shape == (45, 3)
    assert model.predict(X).shape == (45,)
    assert model.n_support_.sum() == len(model.support_)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))


def test_nonlinear_circles():
    X, y = make_circles(n_samples=60, factor=.3, noise=.03, random_state=16)
    model = definitions()['ScratchSVC'](C=10, kernel='rbf', gamma=2).fit(X, y)
    assert np.mean(model.predict(X) == y) > .98


def test_no_free_support_bias():
    X = np.array([[-1.], [1.]])
    y = np.array([-1, 1])
    model = definitions()['BinarySVC'](C=.01).fit(X, y)
    assert np.allclose(model.alpha_, .01)
    assert model.kkt_violation_ < 1e-7
    np.testing.assert_array_equal(model.predict(X), y)


def test_invalid_inputs_and_nonconvergence():
    Binary = definitions()['BinarySVC']
    with pytest.raises(ValueError):
        Binary().fit([[1], [2]], [1, 1])
    with pytest.raises(ValueError):
        Binary(C=-1).fit([[1], [2]], [-1, 1])
    with pytest.raises(RuntimeError, match='optimisation'):
        Binary(max_iter=1).fit([[-2], [-1], [1], [2]], [-1, -1, 1, 1])


def test_shared_preprocessing_and_split():
    from prepare_data import load_data, make_preprocessor, validation_split, class_labels, NUMERIC
    train, test, thresholds = load_data()
    assert len(train) == 313 and len(test) == 79
    assert not set(train.car_id) & set(test.car_id)
    assert train.car_id.is_unique and test.car_id.is_unique
    np.testing.assert_allclose(train.mpg.quantile([1/3, 2/3]), [thresholds['q1'], thresholds['q2']])
    assert list(class_labels([thresholds['q1'], thresholds['q2'], 100], thresholds)) == [0, 1, 2]
    subtrain, validation = validation_split(train, thresholds)
    other, _ = validation_split(train, thresholds)
    np.testing.assert_array_equal(subtrain.car_id, other.car_id)
    processor = make_preprocessor().fit(subtrain)
    means = processor.named_transformers_['numeric'].mean_.copy()
    changed = validation.copy()
    changed[NUMERIC] = 1e9
    changed['origin'] = 'nouvelle origine'
    assert np.isfinite(processor.transform(changed)).all()
    np.testing.assert_allclose(means, subtrain[NUMERIC].mean())
    np.testing.assert_array_equal(means, processor.named_transformers_['numeric'].mean_)


def test_comparison_rejects_invalid_files(tmp_path):
    import pandas as pd
    from compare import load_predictions, align_predictions
    _, test, thresholds = __import__('prepare_data').load_data()
    from prepare_data import class_labels
    frame = pd.DataFrame(dict(car_id=test.car_id, mpg_real=test.mpg,
        classe_reelle=class_labels(test.mpg, thresholds),
        classe_predite=class_labels(test.mpg, thresholds), correct=True, C=1., split='test'))
    path = tmp_path / 'svc.csv'
    with pytest.raises(ValueError, match='absent'):
        load_predictions(path, 'svc_linear', test, thresholds)
    frame.drop(columns='classe_predite').to_csv(path, index=False)
    with pytest.raises(ValueError, match='Colonnes'):
        load_predictions(path, 'svc_linear', test, thresholds)
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='dupliqu'):
        load_predictions(path, 'svc_linear', test, thresholds)
    frame.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match='identifiants'):
        load_predictions(path, 'svc_linear', test, thresholds)
    damaged = frame.copy(); damaged.loc[damaged.index[0], 'mpg_real'] += 1
    damaged.to_csv(path, index=False)
    with pytest.raises(ValueError, match='réelles'):
        load_predictions(path, 'svc_linear', test, thresholds)
    frame.to_csv(path, index=False)
    checked = load_predictions(path, 'svc_linear', test, thresholds)
    joined = align_predictions(checked, checked.sample(frac=1, random_state=16), ['car_id', 'C'])
    assert len(joined) == 79
    incompatible = checked.copy(); incompatible['C'] = 2.
    with pytest.raises(ValueError, match='configurations'):
        align_predictions(checked, incompatible, ['car_id', 'C'])
