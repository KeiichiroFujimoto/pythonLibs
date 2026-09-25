"""Partitioned coupling relaxation: fixed point of a linear contraction, Aitken vs constant factor."""
import numpy as np

from pythonLibs.fieldMapping.Coupling import FixedPointRelaxation


def _iterate(relax, F, x0, n=200):
    x = x0
    for k in range(n):
        x, done = relax.update(x, F(x))
        if done:
            return x, k
    return x, n


def test_converges_to_the_fixed_point_and_aitken_is_faster():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    A = Q @ np.diag(np.linspace(-0.9, 0.6, 20)) @ Q.T           # contraction with an oscillating mode
    b = rng.normal(size=20) * 100.0
    exact = np.linalg.solve(np.eye(20) - A, b)
    F = lambda x: A @ x + b                                      # noqa: E731
    xc, nc = _iterate(FixedPointRelaxation(omega=0.5, aitken=False, tolerance=1e-8), F, np.zeros(20))
    xa, na = _iterate(FixedPointRelaxation(omega=0.5, aitken=True, tolerance=1e-8), F, np.zeros(20))
    np.testing.assert_allclose(xc, exact, atol=1e-6)
    np.testing.assert_allclose(xa, exact, atol=1e-6)
    assert na < nc, (na, nc)


def test_nan_entries_are_left_alone():
    relax = FixedPointRelaxation(omega=1.0, aitken=False, tolerance=0.0)
    x = np.array([1.0, 2.0, 3.0])
    nxt, _ = relax.update(x, np.array([5.0, np.nan, 7.0]))
    np.testing.assert_allclose(nxt, [5.0, 2.0, 7.0])
    assert relax.history[-1]["liveEntries"] == 2
