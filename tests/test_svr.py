"""QP primal, dual, perte epsilon et comparaison contrôlée avec libsvm."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import nbformat
import numpy as np
import pytest
from sklearn.svm import SVR


def definitions():
    namespace = {}
    path = Path(__file__).resolve().parents[1] / 'from_scratch/from_scratch.ipynb'
    for cell in nbformat.read(path, as_version=4).cells:
        if set(cell.metadata.get('tags', [])) & {'scratch_svc', 'scratch_svr'}:
            exec(compile(cell.source, 'from_scratch.ipynb:definitions', 'exec'), namespace)
    return namespace


def test_epsilon_insensitive_loss():
    loss = definitions()['epsilon_insensitive_loss']
    np.testing.assert_allclose(loss([0, 1, 2, -3], [0, 0, 0, 0], 1), [0, 0, 1, 2])


@pytest.mark.parametrize('epsilon', [.1, .5, 1., 2., 5.])
def test_svr_primal_dual_and_reference(epsilon):
    rng = np.random.default_rng(16)
    X = rng.normal(size=(30, 3))
    y = X@np.array([2., -1., .5])+4+rng.normal(scale=.3, size=30)
    model = definitions()['ScratchSVR'](C=3, epsilon=epsilon).fit(X, y)
    reference = SVR(C=3, epsilon=epsilon, kernel='linear', tol=1e-9, shrinking=False).fit(X, y)
    assert model.result_.success
    assert model.predict(X).shape == (30,)
    assert model.constraint_violation_ < 1e-6
    assert model.equality_residual_ < 1e-6
    assert model.stationarity_residual_ < 2e-4
    assert abs(model.duality_gap_) < 1e-4
    assert np.all(model.alpha_ >= -1e-7) and np.all(model.alpha_star_ >= -1e-7)
    assert np.max(model.alpha_+model.alpha_star_) <= 3+1e-6
    np.testing.assert_allclose(model.coef_, X.T@model.dual_coef_, atol=2e-4)
    assert model.n_support_ == len(model.support_)
    np.testing.assert_array_equal(model.support_, np.flatnonzero(model.alpha_+model.alpha_star_ > 1e-5))
    # Si le tube contient tout le train, b peut être non unique ; comparer l'objectif.
    if model.n_support_ > 0:
        np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=2e-3)
    ref_objective = .5*np.sum(reference.coef_**2)+3*definitions()['epsilon_insensitive_loss'](y, reference.predict(X), epsilon).sum()
    assert model.primal_objective_ == pytest.approx(ref_objective, abs=2e-4)


def test_svr_supports_decrease_on_simple_line():
    X = np.linspace(-2, 2, 20)[:, None]
    y = 3*X[:, 0]+2
    small = definitions()['ScratchSVR'](C=2, epsilon=.1).fit(X, y)
    large = definitions()['ScratchSVR'](C=2, epsilon=10).fit(X, y)
    assert small.n_support_ > 0
    assert large.n_support_ == 0
    assert large.primal_objective_ < 1e-6


def test_svr_invalid_inputs_and_failure():
    SVRClass = definitions()['ScratchSVR']
    with pytest.raises(ValueError):
        SVRClass(epsilon=-1).fit([[1], [2]], [1, 2])
    with pytest.raises(ValueError):
        SVRClass().fit([[1], [2]], [1])
    with pytest.raises(ValueError):
        SVRClass().fit([[np.nan], [2]], [1, 2])
    with pytest.raises(RuntimeError, match='optimisation'):
        SVRClass(max_iter=1).fit([[1], [2]], [1, 2])


def test_svr_auto_mpg_feasibility_and_duality():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prepare_data import load_data, make_preprocessor, validation_split
    train, _, thresholds = load_data()
    subtrain, _ = validation_split(train, thresholds)
    X = make_preprocessor().fit_transform(subtrain)
    model = definitions()['ScratchSVR'](C=10, epsilon=.1).fit(X, subtrain.mpg.to_numpy())
    assert model.constraint_violation_ < 1e-6
    assert abs(model.duality_gap_) < 2e-3
    assert model.stationarity_residual_ < 1e-3
