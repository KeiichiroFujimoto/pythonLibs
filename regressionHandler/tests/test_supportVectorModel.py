"""SupportVectorModel (LIBSVM-equivalent epsilon-SVR)."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import SupportVectorModel, loadModel


def _data(n=80, d=2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, (n, d))
    return x, np.sin(x).sum(axis=1) + rng.normal(0.0, 0.1, n)


def test_kktAtSolution():
    x, y = _data()
    m = SupportVectorModel(C=2.0, epsilon=0.1, tol=1e-8).fit(x, y)
    coef = m.dualCoefficients
    assert abs(coef.sum()) < 1e-8 and np.all(np.abs(coef) <= 2.0 + 1e-12)
    resid = y - m.predict(x)
    # points strictly inside the tube carry no weight; free support vectors sit on its edge
    inside = np.abs(resid) < 0.1 - 1e-6
    sv = np.array([any(np.array_equal(p, s) for s in m.supportVectors) for p in x])
    assert not np.any(inside & sv)
    free = sv.copy()
    free[sv] = np.abs(coef) < 2.0 - 1e-9
    assert np.abs(np.abs(resid[free]) - 0.1) == pytest.approx(0.0, abs=1e-6)


def test_linearKernelRecoversLine():
    x = np.linspace(0.0, 1.0, 40)[:, None]
    m = SupportVectorModel(kernel="linear", C=100.0, epsilon=0.01, tol=1e-9).fit(x, 3.0 * x[:, 0] - 1.0)
    assert m.predict(np.array([[0.25], [0.75]])) == pytest.approx([-0.25, 1.25], abs=0.011)


@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
def test_matchesLibsvm(kernel):
    svm = pytest.importorskip("sklearn.svm")
    x, y = _data(n=90, seed=3)
    q = np.random.default_rng(4).uniform(-3.0, 3.0, (100, 2))
    ref = svm.SVR(kernel=kernel, C=1.0, epsilon=0.1, shrinking=False).fit(x, y)
    got = SupportVectorModel(kernel=kernel, C=1.0, epsilon=0.1).fit(x, y)
    # same SMO path up to single-precision kernel rounding; always within the solver tolerance
    assert np.max(np.abs(got.predict(q) - ref.predict(q))) < 1e-3 * np.ptp(y)


def test_smoPathIdenticalToLibsvmUpToGradientRounding():
    """Same kernel matrix -> same iterations and coefficients as LIBSVM, with either separate
    or fused (fma) rounding of the gradient update, whichever the reference build uses."""
    svm = pytest.importorskip("sklearn.svm")
    from fractions import Fraction
    from pythonLibs.regressionHandler.models.SupportVectorModel import solveSvrDual, svrKernel

    def fused(qi, dI, qj, dJ):
        prod = qj * dJ
        return np.array([float(Fraction(a) * Fraction(dI) + Fraction(b)) for a, b in zip(qi, prod)])

    rng = np.random.default_rng(0)
    for _ in range(6):
        n = int(rng.integers(15, 40))
        x = rng.uniform(-3.0, 3.0, (n, 2))
        y = np.sin(x).sum(axis=1) + rng.normal(0.0, 0.2, n)
        k = svrKernel("rbf", x, x, 1.0 / (2 * x.var()), 3, 0.0)
        ref = svm.SVR(kernel="precomputed", C=10.0, epsilon=0.01, shrinking=False).fit(k, y)
        refCoef = np.zeros(n)
        refCoef[ref.support_] = ref.dual_coef_[0]
        matches = []
        for update in (None, fused):
            coef, rho, iters = solveSvrDual(k, y, 10.0, 0.01, 1e-3, 10 ** 6, gradientUpdate=update)
            matches.append(iters == int(np.ravel(ref.n_iter_)[0]) and np.max(np.abs(coef - refCoef)) < 1e-12
                           and abs(rho + ref.intercept_[0]) < 1e-12)
        assert any(matches)


def test_saveLoadRoundTrip(tmp_path):
    x, y = _data()
    m = SupportVectorModel().fit(x, y)
    m.save(str(tmp_path / "svr.json"))
    assert loadModel(str(tmp_path / "svr.json")).predict(x[:5]) == pytest.approx(m.predict(x[:5]), abs=1e-14)
