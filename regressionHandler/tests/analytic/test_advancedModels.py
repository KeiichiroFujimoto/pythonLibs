"""ODR, heteroscedastic / transformed-target models, trees, MLP, mixed models, shape constraints and
Bayesian linear regression vs closed forms and optimality conditions."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import (BayesianLinearModel, GradientBoostingModel, HeteroscedasticModel,
                                          IsotonicModel, LinearBasisModel, MixedModel, NeuralNetworkModel, OdrModel,
                                          RandomForestModel, ShapeSplineModel, TransformedTargetModel)
from pythonLibs.regressionHandler.bases.BSplineBasis import differenceMatrix
from pythonLibs.regressionHandler.models.NeuralNetworkModel import _Net
from pythonLibs.regressionHandler.models.QuantileModel import QuantileModel as QuantileModelCls
from pythonLibs.regressionHandler.models.ShapeConstrainedModels import constrainedLeastSquares, nnls, pava
from pythonLibs.regressionHandler.models.TransformedTargetModel import (boxCox, boxCoxInverse, yeoJohnson,
                                                                       yeoJohnsonInverse)
from pythonLibs.regressionHandler.numerics.Trees import FeatureBinner, buildTree, buildTreeLevelwise


# ---------------------------------------------------------------- orthogonal distance regression
def test_odrLineIsDemingRegression():
    rng = np.random.default_rng(0)
    xt = rng.uniform(0, 10, 300)
    x, y = xt + 0.4 * rng.standard_normal(300), 1 + 2 * xt + 0.8 * rng.standard_normal(300)
    sx, sy = 0.4, 0.8
    m = OdrModel(expression="a+b*x", params=["a", "b"], p0=[0, 1], xSigma=sx, ySigma=sy).fit(x, y)
    lam = (sy / sx) ** 2
    sxx, syy, sxy = np.var(x), np.var(y), np.cov(x, y, bias=True)[0, 1]
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    assert m.parameters["b"] == pytest.approx(b, rel=1e-8)
    assert m.parameters["a"] == pytest.approx(y.mean() - b * x.mean(), rel=1e-7)


def test_odrInnerStationarityAndOlsLimit():
    rng = np.random.default_rng(1)
    xt = rng.uniform(0, 4, 150)
    x, y = xt + 0.1 * rng.standard_normal(150), 2 * np.exp(-0.5 * xt) + 0.05 * rng.standard_normal(150)
    m = OdrModel(expression="a*exp(-b*x)", params=["a", "b"], p0=[1, 1], xSigma=0.1, ySigma=0.05).fit(x, y)
    p = np.array([m.parameters["a"], m.parameters["b"]])
    d = m.inputCorrections[:, 0]
    xd = x + d
    f = p[0] * np.exp(-p[1] * xd)
    g = -p[1] * f
    np.testing.assert_allclose(d / 0.1 ** 2 + (f - y) * g / 0.05 ** 2, 0.0, atol=1e-6)
    tiny = OdrModel(expression="a*exp(-b*x)", params=["a", "b"], p0=[1, 1], xSigma=1e-9, ySigma=1.0).fit(x, y)
    from pythonLibs.regressionHandler import NonlinearModel
    ols = NonlinearModel(expression="a*exp(-b*x)", params=["a", "b"], p0=[1, 1]).fit(x, y)
    for k in ("a", "b"):
        assert tiny.parameters[k] == pytest.approx(ols.parameters[k], rel=1e-7)


# ---------------------------------------------------------------- heteroscedastic / transformed target
def test_heteroscedasticConstantVarianceIsOlsMl():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, (80, 1))
    y = 1 + 3 * x[:, 0] + 0.2 * rng.standard_normal(80)
    h = HeteroscedasticModel(varianceBasis={"type": "polynomial", "degree": 0}).fit(x, y)
    ols = LinearBasisModel().fit(x, y)
    np.testing.assert_allclose(h._beta, ols.coefficients.ravel(), rtol=1e-10)
    rss = float(np.sum((y - ols.predict(x).ravel()) ** 2))
    assert h.predictStd(x[:1])[0] ** 2 == pytest.approx(rss / 80, rel=1e-8)


def test_heteroscedasticIsLikelihoodStationary():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 2, (300, 1))
    y = 1 + 2 * x[:, 0] + np.exp(0.5 * (-1 + x[:, 0])) * rng.standard_normal(300)
    h = HeteroscedasticModel().fit(x, y)
    phi = np.column_stack([np.ones(300), x[:, 0]])
    theta = np.concatenate([h._beta, h._gamma])

    def ll(t):
        s2 = np.exp(phi @ t[2:])
        return -0.5 * np.sum(np.log(2 * np.pi * s2) + (y - phi @ t[:2]) ** 2 / s2)
    grad = [(ll(theta + 1e-6 * e) - ll(theta - 1e-6 * e)) / 2e-6 for e in np.eye(4)]
    np.testing.assert_allclose(grad, 0.0, atol=1e-4)
    assert h.varianceParameters["logLikelihood"] == pytest.approx(ll(theta), rel=1e-10)


def test_transformsInvertAndIdentity():
    y = np.array([0.1, 0.7, 2.0, 9.0])
    for lam in (-1.0, 0.0, 0.5, 2.0):
        np.testing.assert_allclose(boxCoxInverse(boxCox(y, lam), lam), y, rtol=1e-12)
    ys = np.array([-3.0, -0.5, 0.0, 0.4, 5.0])
    for lam in (-0.5, 0.0, 1.0, 2.0, 2.7):
        np.testing.assert_allclose(yeoJohnsonInverse(yeoJohnson(ys, lam), lam), ys, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(yeoJohnson(ys, 1.0), ys, atol=1e-15)


def test_transformedTargetLogAndProfile():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, (100, 1))
    y = np.exp(1 + 2 * x[:, 0] + 0.3 * rng.standard_normal(100))
    m = TransformedTargetModel(model="linear", transform="log").fit(x, y)
    inner = LinearBasisModel().fit(x, np.log(y))
    t = np.array([[0.2], [0.8]])
    np.testing.assert_allclose(m.predict(t).ravel(), np.exp(inner.predict(t).ravel()), rtol=1e-12)
    band, ib = m.predictInterval(t, 0.9), inner.predictInterval(t, 0.9)
    np.testing.assert_allclose(band.lower.ravel(), np.exp(ib.lower.ravel()), rtol=1e-12)
    bc = TransformedTargetModel(model="linear").fit(x, y)
    lam = bc.transformParameter
    assert bc._profile(lam) >= max(bc._profile(lam - 0.05), bc._profile(lam + 0.05))
    none = TransformedTargetModel(model="linear", transform="none").fit(x, y)
    np.testing.assert_allclose(none.predict(t), LinearBasisModel().fit(x, y).predict(t), rtol=1e-12)


# ---------------------------------------------------------------- trees and ensembles
def test_stumpIsExhaustiveBestSplit():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, (60, 1))
    y = np.where(x[:, 0] > 0.37, 2.0, 0.0) + 0.3 * rng.standard_normal(60)
    b = FeatureBinner(255).fit(x)
    tree = buildTreeLevelwise(b.transform(x), b, -y, np.ones(60), maxDepth=1)
    xs = np.sort(x[:, 0])
    best = min(((np.sum((y[x[:, 0] <= t] - y[x[:, 0] <= t].mean()) ** 2) +
                 np.sum((y[x[:, 0] > t] - y[x[:, 0] > t].mean()) ** 2)), t) for t in 0.5 * (xs[1:] + xs[:-1]))
    assert tree.threshold[0] == pytest.approx(best[1])
    left = x[:, 0] <= best[1]
    np.testing.assert_allclose(sorted(tree.value[1:]), sorted([y[left].mean(), y[~left].mean()]), rtol=1e-12)


def test_fullTreesInterpolateAndBuildersAgree():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 1, (80, 3))
    y = np.sin(4 * x[:, 0]) + x[:, 1]
    b = FeatureBinner(255).fit(x)
    c = b.transform(x)
    lw = buildTreeLevelwise(c, b, -y, np.ones(80))
    bf = buildTree(c, b, -y, np.ones(80))
    # both builders grow until the leaves are pure; tied gains may pick different splits off-sample
    np.testing.assert_allclose(lw.predict(x), y, atol=1e-12)
    np.testing.assert_allclose(bf.predict(x), y, atol=1e-12)


def test_forestOutOfBagAndBoostingProperties():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, (300, 3))
    y = 3 * x[:, 0] + np.sin(5 * x[:, 1]) + 0.2 * rng.standard_normal(300)
    rf = RandomForestModel(nTrees=30).fit(x, y)
    preds = np.array([t.predict(x) for t in rf._trees])
    oob = rf._inbag == 0
    manual = np.sum(preds * oob, 0) / np.maximum(oob.sum(0), 1)
    ok = oob.sum(0) > 0
    np.testing.assert_allclose(rf.outOfBag["predictions"][ok], manual[ok], rtol=1e-12)
    assert np.all(rf.predictVariances(x[:10]) >= 0)
    stump = GradientBoostingModel(nEstimators=1, learningRate=1.0, maxDepth=1, maxLeaves=2, l2=0.0,
                                  validationFraction=0.0, minSamplesLeaf=1).fit(x, y)
    b = FeatureBinner(255).fit(x)
    ref = buildTreeLevelwise(b.transform(x), b, -(y - y.mean()), np.ones(300), maxDepth=1)
    np.testing.assert_allclose(stump.predict(x).ravel(), y.mean() + ref.predict(x), rtol=1e-10)
    q = GradientBoostingModel(loss="quantile", tau=0.8, validationFraction=0.0, nEstimators=300).fit(x, y)
    assert abs(np.mean(y <= q.predict(x).ravel()) - 0.8) < 0.05


# ---------------------------------------------------------------- neural network
def test_mlpGradientsAndLinearLimit():
    net = _Net([3, 6, 4, 2], "tanh")
    rng = np.random.default_rng(8)
    th, x, y = net.init(rng), rng.standard_normal((9, 3)), rng.standard_normal((9, 2))
    _, g = net.lossGrad(th, x, y, 0.3, 9)
    num = [(net.lossGrad(th + 1e-6 * e, x, y, 0.3, 9)[0] - net.lossGrad(th - 1e-6 * e, x, y, 0.3, 9)[0]) / 2e-6
           for e in np.eye(th.size)]
    np.testing.assert_allclose(g, num, atol=1e-8)
    d = np.array([0.3, -1.0, 0.5])
    np.testing.assert_allclose(net.tangent(th, x, d), (net.predict(th, x + 1e-6 * d) - net.predict(th, x - 1e-6 * d)) / 2e-6,
                               atol=1e-8)
    xx = rng.uniform(-1, 1, (50, 2))
    yy = 1 + xx @ [2.0, -3.0] + 0.1 * rng.standard_normal(50)
    lin = NeuralNetworkModel(hidden=[], weightDecay=0.0).fit(xx, yy)
    np.testing.assert_allclose(lin.predict(xx), LinearBasisModel().fit(xx, yy).predict(xx), atol=1e-6)


# ---------------------------------------------------------------- mixed model
def test_mixedModelMatchesDenseFormulas():
    rng = np.random.default_rng(9)
    g = np.repeat(np.arange(12), rng.integers(3, 8, 12))
    n = g.size
    x = rng.uniform(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, 0.8, 12)[g] + rng.normal(0, 0.4, 12)[g] * x + 0.3 * rng.standard_normal(n)
    m = MixedModel(randomSlopes=[0]).fit(np.column_stack([x, g]), y)
    X = np.column_stack([np.ones(n), x])
    Z = np.zeros((n, 24))
    Z[np.arange(n), 2 * g] = 1.0
    Z[np.arange(n), 2 * g + 1] = x
    psi = np.array(m.varianceComponents()["covariance"])
    V = m._sigma2 * np.eye(n) + Z @ np.kron(np.eye(12), psi) @ Z.T
    Vi = np.linalg.inv(V)
    beta = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
    np.testing.assert_allclose(m._beta, beta, rtol=1e-8)
    r = y - X @ beta
    reml = -0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(X.T @ Vi @ X)[1] + r @ Vi @ r + (n - 2) * np.log(2 * np.pi))
    assert m.varianceComponents()["logLikelihood"] == pytest.approx(reml, rel=1e-8)
    blup = np.kron(np.eye(12), psi) @ Z.T @ Vi @ r
    np.testing.assert_allclose(m._blup.ravel(), blup, atol=1e-8)
    np.testing.assert_allclose(m._covBeta, np.linalg.inv(X.T @ Vi @ X), rtol=1e-8)


# ---------------------------------------------------------------- shape constraints
def test_nnlsKktAndConstrainedLeastSquares():
    rng = np.random.default_rng(10)
    a, b = rng.standard_normal((25, 8)), rng.standard_normal(25)
    x = nnls(a, b)
    grad = a.T @ (a @ x - b)
    assert np.all(x >= 0)
    np.testing.assert_allclose(grad[x > 0], 0.0, atol=1e-10)
    assert np.all(grad[x == 0] >= -1e-10)
    m, r = rng.standard_normal((30, 6)), rng.standard_normal(30)
    d = differenceMatrix(6, 1)
    c = constrainedLeastSquares(m, r, d)
    assert np.all(d @ c >= -1e-10)
    free = np.linalg.lstsq(m, r, rcond=None)[0]
    if np.all(d @ free >= 0):
        np.testing.assert_allclose(c, free)
    # KKT: M^T (M c - r) = A^T mu with mu >= 0 on active rows
    active = np.abs(d @ c) < 1e-9
    mu = np.linalg.lstsq(d[active].T, m.T @ (m @ c - r), rcond=None)[0] if active.any() else np.zeros(0)
    np.testing.assert_allclose(d[active].T @ mu, m.T @ (m @ c - r), atol=1e-8)
    assert np.all(mu >= -1e-8)


def test_pavaAndIsotonicModel():
    np.testing.assert_allclose(pava(np.array([1.0, 3.0, 2.0, 4.0, 3.5, 5.0])), [1, 2.5, 2.5, 3.75, 3.75, 5])
    np.testing.assert_allclose(pava(np.array([3.0, 1.0]), np.array([1.0, 3.0])), [1.5, 1.5])
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 1, 100)
    y = np.log1p(4 * x) + 0.2 * rng.standard_normal(100)
    m = IsotonicModel().fit(x, y)
    fit = m.predict(np.sort(x)).ravel()
    assert np.all(np.diff(fit) >= -1e-12)
    assert np.sum(fit) == pytest.approx(np.sum(y), rel=1e-12)          # blocks preserve means
    mono = np.sort(y)
    np.testing.assert_allclose(IsotonicModel().fit(np.sort(x), mono).predict(np.sort(x)).ravel(), mono)


@pytest.mark.parametrize("shape", ["increasing", "decreasing", "convex", "concave", "increasingConcave",
                                   "decreasingConvex"])
def test_shapeSplineSatisfiesShape(shape):
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 1, 150)
    y = np.sin(3 * x) + 0.2 * rng.standard_normal(150)
    m = ShapeSplineModel(shape=shape).fit(x, y)
    c = m._coef
    if shape.startswith("increasing"):
        assert np.all(np.diff(c) >= -1e-10)
    if shape.startswith("decreasing"):
        assert np.all(np.diff(c) <= 1e-10)
    if shape.lower().endswith("convex"):
        assert np.all(np.diff(c, 2) >= -1e-10)
    if shape.lower().endswith("concave"):
        assert np.all(np.diff(c, 2) <= 1e-10)
    grid = np.linspace(x.min(), x.max(), 300)
    f = m.predict(grid).ravel()
    if shape.startswith("increasing"):
        assert np.all(np.diff(f) >= -1e-10)
    if shape.lower().endswith("concave"):
        assert np.all(np.diff(f, 2) <= 1e-10)


# ---------------------------------------------------------------- Bayesian linear regression
def test_bayesianRidgeFixedPointAndPosterior():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((80, 4))
    y = x @ [1.0, -2.0, 0.0, 0.5] + 0.5 + 0.3 * rng.standard_normal(80)
    m = BayesianLinearModel().fit(x, y)
    phi = np.column_stack([np.ones(80), x])
    ev = m.evidence
    alpha = np.array([0.0] + ev["alpha"][1:])
    beta = 1.0 / ev["noiseVariance"]
    post = np.linalg.solve(np.diag(alpha) + beta * phi.T @ phi, beta * phi.T @ y)
    np.testing.assert_allclose(m._mean, post, rtol=1e-6)
    sigma = np.linalg.inv(np.diag(alpha) + beta * phi.T @ phi)
    np.testing.assert_allclose(m._cov, sigma, rtol=1e-6)
    gamma = 1 - alpha * np.diag(sigma)
    a = alpha[1]
    assert a == pytest.approx(np.sum(gamma[1:]) / np.sum(post[1:] ** 2), rel=1e-5)
    assert beta == pytest.approx((80 - np.sum(gamma)) / np.sum((y - phi @ post) ** 2), rel=1e-5)
    ard = BayesianLinearModel(prior="ard").fit(x, y)
    assert "x2" not in ard.evidence["activeTerms"] and {"x0", "x1", "x3"} <= set(ard.evidence["activeTerms"])


# ---------------------------------------------------------------- persistence encoding, vertices, forests
def test_compactEncodingIsLosslessAndSmaller():
    import json
    from pythonLibs.regressionHandler.core.Serialization import compactArrays, expandArrays
    rng = np.random.default_rng(14)
    data = {"f": rng.standard_normal(500).tolist(), "i": list(range(300)), "m": rng.standard_normal((40, 7)).tolist(),
            "n": [None if k % 7 == 0 else float(k) / 3 for k in range(200)], "s": ["a", "b"], "small": [1.5, 2.5],
            "nested": [{"v": rng.standard_normal(100).tolist()}]}
    enc = compactArrays(data)
    assert enc["small"] == [1.5, 2.5] and enc["s"] == ["a", "b"]
    assert expandArrays(json.loads(json.dumps(enc))) == data
    assert len(json.dumps(enc)) < 0.7 * len(json.dumps(data))
    x = rng.uniform(0, 1, (300, 2))
    y = x[:, 0] + np.sin(5 * x[:, 1])
    m = RandomForestModel(nTrees=10).fit(x, y)
    from pythonLibs.regressionHandler import SurrogateModelBase
    for compact in (False, True):
        d = json.loads(json.dumps(m.toDict(compact=compact)))
        np.testing.assert_array_equal(SurrogateModelBase.fromDict(d).predict(x), m.predict(x))


def test_quantileSolutionIsAVertex():
    rng = np.random.default_rng(15)
    x = rng.uniform(0, 1, (400, 3))
    y = x @ [1.0, -2.0, 0.5] + rng.standard_t(3, 400)
    m = QuantileModelCls(tau=0.35, se="none").fit(x, y)
    r = y - m.predict(x).ravel()
    assert np.sum(np.abs(r) < 1e-10) == 4                      # p = 4 interpolated observations


def test_forestBatchEqualsSingleTrees():
    from pythonLibs.regressionHandler.numerics.Trees import buildForestLevelwise
    rng = np.random.default_rng(16)
    x = rng.uniform(0, 1, (200, 3))
    # integer responses keep every gradient sum exact, so tie-breaking cannot depend on rounding
    # (with real-valued data, near-tied splits may resolve differently between batch and single builds)
    y = rng.integers(-5, 6, 200).astype(float) + np.round(3 * x[:, 0])
    b = FeatureBinner(255).fit(x)
    c = b.transform(x)
    sets = [rng.integers(0, 200, 200) for _ in range(4)]
    forest = buildForestLevelwise(c, b, -y, np.ones(200), sets, minSamplesLeaf=3)
    for rows, tree in zip(sets, forest):
        single = buildTreeLevelwise(c, b, -y, np.ones(200), rows, minSamplesLeaf=3)
        np.testing.assert_array_equal(tree.feature, single.feature)
        np.testing.assert_allclose(tree.predict(x), single.predict(x), rtol=0, atol=0)


def test_bsplineRangeExtension():
    from pythonLibs.regressionHandler.bases.BSplineBasis import BSplineBasis
    x = np.linspace(2.0, 4.0, 30)[:, None]
    b = BSplineBasis(nSegments=5, rangeExtension=0.01).fit(x)
    assert b._lo[0] == pytest.approx(1.98) and b._hi[0] == pytest.approx(4.02)
    np.testing.assert_allclose(b.transform(x).sum(axis=1), 1.0, atol=1e-12)     # partition of unity
