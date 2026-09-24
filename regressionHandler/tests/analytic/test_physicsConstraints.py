"""Physics-preserving regression and conservative projection against closed-form solutions."""
import itertools

import numpy as np
import pytest

from pythonLibs.regressionHandler import (ConstrainedModel, DimensionlessModel, KrigingModel, LinearBasisModel,
                                          RegressionHandler)
from pythonLibs.regressionHandler.constraints import (boundedQp, boxQuadrature, buildConstraints, feasibility,
                                                      gaussLegendre, projectAffine, projectNonlinear)
from pythonLibs.regressionHandler.constraints.DimensionalAnalysis import (buckinghamPi, dimensionMatrix,
                                                                          scalingExponents)
import pythonLibs.regressionHandler.numerics.Sparse as sparseModule
from pythonLibs.regressionHandler.numerics.Sparse import SparseMatrix


# ---------------------------------------------------------------- building blocks
def test_sparseMatrixMatchesDense():
    rng = np.random.default_rng(0)
    a = np.where(rng.uniform(size=(7, 40)) < 0.2, rng.normal(size=(7, 40)), 0.0)
    s = SparseMatrix.fromDense(a)
    x, y, v = rng.normal(size=40), rng.normal(size=7), rng.uniform(0.5, 2, 40)
    np.testing.assert_allclose(s @ x, a @ x, atol=1e-14)
    np.testing.assert_allclose(s.rmatvec(y), a.T @ y, atol=1e-14)
    np.testing.assert_allclose(s.gram(v), (a * v) @ a.T, atol=1e-13)
    old = sparseModule.DENSE_GRAM_LIMIT
    try:
        sparseModule.DENSE_GRAM_LIMIT = 0                         # pairwise path
        np.testing.assert_allclose(s.gram(v), (a * v) @ a.T, atol=1e-13)
    finally:
        sparseModule.DENSE_GRAM_LIMIT = old
    dup = SparseMatrix([0, 0, 1], [2, 2, 0], [1.0, 2.0, 5.0], (2, 3))
    np.testing.assert_array_equal(dup.toDense(), [[0, 0, 3], [5, 0, 0]])


def test_boundedQpIsTheBruteForceOptimum():
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 5
        q = rng.normal(size=(n, n))
        h, g = q @ q.T + 0.1 * np.eye(n), rng.normal(size=n)
        lb = np.where(rng.uniform(size=n) < 0.5, -0.2, -np.inf)
        ub = np.where(rng.uniform(size=n) < 0.5, 0.3, np.inf)
        x = boundedQp(h, g, lb, ub, ridge=0.0)
        best = np.inf
        for state in itertools.product((0, 1, 2), repeat=n):    # free / at lower / at upper
            fixed = np.array(state)
            if np.any((fixed == 1) & ~np.isfinite(lb)) or np.any((fixed == 2) & ~np.isfinite(ub)):
                continue
            z = np.where(fixed == 1, lb, np.where(fixed == 2, ub, 0.0))
            f = fixed == 0
            if f.any():
                z[f] = np.linalg.solve(h[np.ix_(f, f)], g[f] - h[np.ix_(f, ~f)] @ z[~f])
            if np.all(z >= lb - 1e-12) and np.all(z <= ub + 1e-12):
                best = min(best, 0.5 * z @ h @ z - g @ z)
        assert 0.5 * x @ h @ x - g @ x == pytest.approx(best, abs=1e-10)


def test_projectAffineClosedFormAndKkt():
    rng = np.random.default_rng(2)
    n, m = 60, 4
    x0, w = rng.normal(size=n), rng.uniform(0.5, 3, n)
    c, d = rng.normal(size=(m, n)), rng.normal(size=m)
    r = projectAffine(x0, w, c, d)
    wi = 1 / w
    exact = x0 - wi * (c.T @ np.linalg.solve((c * wi) @ c.T, c @ x0 - d))
    np.testing.assert_allclose(r.x, exact, atol=1e-12)
    assert r.iterations == 1
    rb = projectAffine(x0, w, c, d, lb=-0.8, ub=0.9)
    assert rb.converged
    np.testing.assert_allclose(c @ rb.x, d, atol=1e-11)
    np.testing.assert_allclose(rb.x, np.clip(x0 - wi * (c.T @ rb.multipliers), -0.8, 0.9), atol=1e-12)


def test_projectAffineLargeGroupsAndInfeasibility():
    rng = np.random.default_rng(3)
    n, m = 200_000, 50
    x0, w = rng.normal(1.0, 1.0, n), rng.uniform(0.5, 2, n)
    grp = rng.integers(0, m, n)
    c = SparseMatrix(grp, np.arange(n), rng.uniform(0.5, 1.5, n), (m, n))
    d = c @ np.abs(x0) * 1.05
    r = projectAffine(x0, w, c, d, lb=0.0)
    assert r.converged and r.x.min() >= 0.0
    np.testing.assert_allclose(c @ r.x, d, rtol=1e-12)
    rep = feasibility(c, -np.ones(m), 0.0, np.inf)
    assert not rep["feasible"]
    assert not projectAffine(x0, w, c, -np.ones(m), lb=0.0).converged


def test_projectNonlinearEnergyConstraint():
    rng = np.random.default_rng(4)
    n = 80
    a, x0 = rng.uniform(1, 2, n), rng.uniform(300, 1500, n)
    grp = np.arange(n) % 4

    def energy(x):
        e = x + 2e-4 * x ** 2 + 1e-8 * x ** 3
        de = 1 + 4e-4 * x + 3e-8 * x ** 2
        return np.bincount(grp, a * e, 4), SparseMatrix(grp, np.arange(n), a * de, (4, n))
    target = energy(x0 * 1.07)[0]
    r = projectNonlinear(x0, a, energy, target, lb=280.0)
    assert r.converged and r.iterations <= 6
    np.testing.assert_allclose(energy(r.x)[0], target, rtol=1e-12)


def test_quadratureIsExact():
    t, w = gaussLegendre(5)
    for k in range(10):
        assert np.sum(w * t ** k) == pytest.approx((1 - (-1) ** (k + 1)) / (k + 1), abs=1e-14)
    p, w = boxQuadrature([0, 1], [2, 4], 3)
    assert np.sum(w * p[:, 0] ** 3 * p[:, 1] ** 2) == pytest.approx(4.0 * (64 - 1) / 3, rel=1e-13)


# ---------------------------------------------------------------- constrained regression
def _decay(n=30, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    return x, np.exp(-3 * x) + 0.05 * rng.standard_normal(n)


TOTAL = (1 - np.exp(-3)) / 3


def test_linearRouteIsConstrainedLeastSquares():
    x, y = _decay()
    w = np.random.default_rng(5).uniform(0.5, 2, x.size)
    cons = [{"type": "integral", "value": TOTAL, "box": {"lower": [0], "upper": [1]}},
            {"type": "value", "points": [[0.0]], "values": [1.0]},
            {"type": "derivative", "points": [[1.0]], "kx": 0, "values": [-0.15]}]
    m = ConstrainedModel(model="poly4", constraints=cons).fit(x[:, None], y, weights=w)
    assert m._route == "linear"
    xv = np.vander(x, 5, increasing=True)
    qp, qw = boxQuadrature([0], [1], 8)
    a = np.vstack([qw @ np.vander(qp[:, 0], 5, increasing=True), [1, 0, 0, 0, 0], [0, 1, 2, 3, 4]])
    c = np.array([TOTAL, 1.0, -0.15])
    k = np.block([[xv.T @ (w[:, None] * xv), a.T], [a, np.zeros((3, 3))]])
    beta = np.linalg.solve(k, np.concatenate([xv.T @ (w * y), c]))[:5]
    np.testing.assert_allclose(m.coefficients[:, 0], beta, atol=1e-10)
    # coefficient covariance projected on the constraints: Cov - Cov A^T (A Cov A^T)^-1 A Cov
    cov = m.baseModel.result.covariance[0]
    proj = cov - cov @ a.T @ np.linalg.solve(a @ cov @ a.T, a @ cov)
    np.testing.assert_allclose(m.result.covariance[0], proj, atol=1e-12 * np.abs(cov).max())
    assert m.predictVariances([[0.0]])[0, 0] == pytest.approx(0.0, abs=1e-14)
    assert m.predictDerivatives([[1.0]], 0)[0, 0] == pytest.approx(-0.15, abs=1e-10)


def test_inequalitiesSatisfyKkt():
    x, y = _decay(seed=1)
    grid = np.linspace(0, 1, 41)[:, None]
    cons = [{"type": "integral", "value": TOTAL, "box": {"lower": [0], "upper": [1]}},
            {"type": "bound", "points": grid, "lower": 0.0},
            {"type": "monotone", "points": grid, "kx": 0, "increasing": False}]
    m = ConstrainedModel(model="poly5", constraints=cons).fit(x[:, None], y)
    rep = m.constraintReport()
    tol = 1e-10
    for r in rep:
        if r["kind"] == "lower":
            assert r["residual"] >= -tol and r["multiplier"] >= -tol
        elif r["kind"] == "upper":
            assert r["residual"] <= tol and r["multiplier"] <= tol
        else:
            assert abs(r["residual"]) <= tol
        if r["kind"] != "equal":                                  # complementary slackness
            assert abs(r["residual"] * r["multiplier"]) <= 1e-9
    assert np.all(m.predictDerivatives(grid, 0) <= 1e-10)               # exact derivatives on the grid


def test_gaussianRouteIsPosteriorConditioning():
    x, y = _decay(20, seed=2)
    cons = [{"type": "mean", "value": TOTAL, "box": {"lower": [0], "upper": [1], "n": 6}},
            {"type": "value", "points": [[0.0]], "values": [1.0]}]
    m = ConstrainedModel(model={"type": "kriging", "corr": "matern52"}, constraints=cons).fit(x[:, None], y)
    assert m._route == "gaussian"
    base = m.baseModel
    qp, qw = boxQuadrature([0], [1], 6)
    pts = np.vstack([qp, [[0.0]]])
    lmat = np.zeros((2, pts.shape[0]))
    lmat[0, :6] = qw / qw.sum()
    lmat[1, 6] = 1.0
    xs = np.array([[0.1], [0.45], [0.9]])
    allp = np.vstack([xs, pts])
    mu = base.predictValues(allp)[:, 0]
    cov = base.predictCovariance(allp)[0]
    sxp, spp = cov[:3, 3:], cov[3:, 3:]
    h = lmat @ spp @ lmat.T
    gain = sxp @ lmat.T @ np.linalg.inv(h)
    mean = mu[:3] + gain @ (np.array([TOTAL, 1.0]) - lmat @ mu[3:])
    var = np.diag(cov[:3, :3] - gain @ lmat @ sxp.T)
    np.testing.assert_allclose(m.predict(xs).ravel(), mean, atol=1e-9)
    np.testing.assert_allclose(m.predictVariances(xs).ravel(), var, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(m.predictCovariance(xs)[0].diagonal(), var, rtol=1e-6, atol=1e-12)


def test_outputBalanceIsWeightedByOutputNoise():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 1, 40)
    frac = np.column_stack([0.2 + 0.3 * x, 0.5 - 0.1 * x, 0.3 - 0.2 * x])
    y = frac + rng.normal(size=(40, 3)) * np.array([0.01, 0.05, 0.02])
    pts = np.linspace(0, 1, 5)[:, None]
    m = ConstrainedModel(model="linear", constraints=[{"type": "outputSum", "points": pts,
                                                        "coefficients": [1, 1, 1], "value": 1.0}]).fit(x[:, None], y)
    grid = np.linspace(-0.5, 1.5, 7)[:, None]
    np.testing.assert_allclose(m.predict(grid).sum(axis=1), 1.0, atol=1e-12)   # linear: holds everywhere
    # closed form: min sum_j |y_j - X b_j|^2 / s_j^2  s.t. sum_j b_j = (1, 0)
    xv = np.column_stack([np.ones(40), x])
    s2 = m.baseModel.result.sigma2
    big = np.zeros((6, 6))
    rhs = np.zeros(6)
    for j in range(3):
        big[2 * j:2 * j + 2, 2 * j:2 * j + 2] = xv.T @ xv / s2[j]
        rhs[2 * j:2 * j + 2] = xv.T @ y[:, j] / s2[j]
    a = np.hstack([np.eye(2)] * 3)
    k = np.block([[big, a.T], [a, np.zeros((2, 2))]])
    sol = np.linalg.solve(k, np.concatenate([rhs, [1.0, 0.0]]))[:6]
    np.testing.assert_allclose(m.coefficients.T.ravel(), sol, atol=1e-10)


def test_kernelRouteAndPersistence():
    x, y = _decay(seed=3)
    cons = [{"type": "value", "points": [[0.0], [1.0]], "values": [1.0, np.exp(-3)]}]
    m = ConstrainedModel(model={"type": "randomForest", "nTrees": 30}, constraints=cons).fit(x[:, None], y)
    assert m._route == "kernel"
    np.testing.assert_allclose(m.predict([[0.0], [1.0]]).ravel(), [1.0, np.exp(-3)], atol=1e-10)
    for model in (m, ConstrainedModel(model="quadratic", constraints=cons).fit(x[:, None], y),
                  ConstrainedModel(model="kriging", constraints=cons).fit(x[:, None], y)):
        back = ConstrainedModel.fromDict(model.toDict())
        np.testing.assert_allclose(back.predict(x[:5, None]), model.predict(x[:5, None]), rtol=1e-12)


def test_constraintSpecsAndHandler():
    specs = [{"type": "integral", "value": 1.0, "box": {"lower": [0, 0], "upper": [1, 2]}, "n": 3},
             {"type": "convex", "points": [[0.5, 0.5]], "kx": 1}]
    cons = buildConstraints(specs)
    assert len(cons) == 2 and cons[0].terms[0].weights.sum() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        buildConstraints([{"type": "unknown"}])
    rh = RegressionHandler()
    x, y = _decay()
    rh.setData(x=x.tolist(), y=y.tolist(), dataName="d")
    spec = {"type": "constrained", "model": "poly3",
            "constraints": [{"type": "integral", "value": TOTAL, "box": {"lower": [0], "upper": [1]}}]}
    rh.fitModel("c", spec, "d")
    rep = rh.inspectModel("c", "constraintReport")["value"]
    assert abs(rep[0]["residual"]) < 1e-12


# ---------------------------------------------------------------- dimensional analysis
def test_buckinghamGroupsAreDimensionless():
    # density, viscosity, velocity, diameter, pressure gradient
    dims = [{"M": 1, "L": -3}, {"M": 1, "L": -1, "T": -1}, {"L": 1, "T": -1}, {"L": 1}, {"M": 1, "L": -2, "T": -2}]
    d = dimensionMatrix(dims)
    pi = buckinghamPi(d)
    assert pi["groups"].shape == (2, 5) and pi["rank"] == 3
    np.testing.assert_array_equal(d @ pi["groups"].T, 0)
    b = scalingExponents(d, [1, 1, -2], pi["repeating"])        # a stress: M L^-1 T^-2
    np.testing.assert_allclose(d @ b, [1, 1, -2])


def test_dimensionlessModelRecoversPhysicalLaws():
    rng = np.random.default_rng(7)
    length, grav, mass = rng.uniform(0.1, 3, 40), rng.uniform(1, 20, 40), rng.uniform(0.1, 5, 40)
    period = 2 * np.pi * np.sqrt(length / grav)
    pend = DimensionlessModel(inputDimensions=[{"L": 1}, {"L": 1, "T": -2}, {"M": 1}], outputDimension={"T": 1})
    pend.fit(np.column_stack([length, grav, mass]), period)
    assert pend._const == pytest.approx(2 * np.pi, rel=1e-12)
    # unit change (m -> cm): same period
    assert np.ravel(pend.predict([[100.0, 981.0, 1.0]]))[0] == pytest.approx(np.ravel(pend.predict([[1.0, 9.81, 1.0]]))[0])
    # smooth-pipe friction: f = 0.316 Re^-0.25 with Re = rho V D / mu
    rho, mu, vel, dia = rng.uniform(800, 1200, 60), rng.uniform(1e-3, 5e-3, 60), rng.uniform(1, 5, 60), \
        rng.uniform(0.02, 0.2, 60)
    re = rho * vel * dia / mu
    dpdl = 0.316 * re ** -0.25 * rho * vel ** 2 / (2 * dia)
    dims = [{"M": 1, "L": -3}, {"M": 1, "L": -1, "T": -1}, {"L": 1, "T": -1}, {"L": 1}]
    # repeating rho, V, D: the output scale is rho V^2 / D and the single group is Re^(+-1)
    m = DimensionlessModel(inputDimensions=dims, outputDimension={"M": 1, "L": -2, "T": -2}, logOutput=True,
                           repeating=[0, 2, 3], model="linear").fit(np.column_stack([rho, mu, vel, dia]), dpdl)
    np.testing.assert_allclose(np.abs(m._groups[0]), [1, 1, 1, 1])
    np.testing.assert_allclose(m._scale, [1, 0, 2, -1])
    assert abs(m.baseModel.coefficients[1, 0]) == pytest.approx(0.25, abs=1e-10)
    np.testing.assert_allclose(m.predict(np.column_stack([rho, mu, vel, dia])[:5]).ravel(), dpdl[:5], rtol=1e-10)
