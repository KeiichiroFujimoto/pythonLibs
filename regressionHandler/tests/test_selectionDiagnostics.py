"""Model selection, stepwise, tuning, diagnostics, bootstrap, problems, reports and handler commands."""
import numpy as np

from pythonLibs.regressionHandler import (ModelSelector, RegressionHandler, bootstrap, createModel, defaultCandidates,
                                          diagnose, getProblem, modelReport, stepwiseSelect, tuneHyperparameters)
from pythonLibs.regressionHandler.evaluation.Diagnostics import runsTest


def test_diagnosticsFlagMisspecificationAndHeteroscedasticity():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 80)
    rep = diagnose(createModel("linear").fit(x, 1 + 3 * x ** 2 + rng.normal(0, 0.02, 80)))
    assert rep.outputs[0]["runsP"] < 1e-3
    assert any("cluster" in w for w in rep.warnings)
    y = 1 + x + rng.normal(0, 0.02 + 0.5 * x, 80)
    rep = diagnose(createModel("linear").fit(x, y))
    assert rep.outputs[0]["breuschPaganP"] < 0.01
    assert "Breusch-Pagan" in rep.summary()
    z, p = runsTest(np.array([1, -1] * 20, dtype=float), np.arange(40))
    assert z > 0 and p < 1e-6


def test_selectorPicksGpOnBranin():
    prob = getProblem("branin")
    x, y = prob.sample(40, seed=0)
    sel = ModelSelector(defaultCandidates(x, y), nJobs=2).run(x, y)
    assert sel.best.name == "gp"
    xt, yt = prob.sample(300, seed=9)
    assert np.sqrt(np.mean((sel.bestModel.predict(xt) - yt) ** 2)) / np.std(yt) < 0.03
    assert "best: gp" in sel.summary()
    failing = ModelSelector(["linear", {"type": "linearBasis", "basis": {"type": "bspline", "maxInputs": 1},
                                        "name": "badSpline"}]).run(x, y)
    assert any(s.error for s in failing.scores) and failing.best.name == "linear"


def test_oneStandardErrorPrefersSimplerModel():
    rng = np.random.default_rng(2)
    x = np.linspace(-1, 1, 40)
    y = 1 + 2 * x + rng.normal(0, 0.3, 40)
    sel = ModelSelector(["poly5", "poly3", "linear"], oneStandardError=True, nFolds=10).run(x, y)
    assert sel.best.name == "linear"


def test_informationCriteriaSelection():
    rng = np.random.default_rng(3)
    x = np.linspace(-1, 1, 60)
    y = 1 + x - 2 * x ** 2 + rng.normal(0, 0.05, 60)
    for crit in ("aicc", "bic"):
        assert ModelSelector(["linear", "quadratic", "poly6"], criterion=crit).run(x, y).best.name == "quadratic"


def test_stepwiseRecoversTrueTerms():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (120, 3))
    y = 1 + 2 * x[:, 0] - x[:, 1] * x[:, 2] + 0.5 * x[:, 2] ** 2 + rng.normal(0, 0.05, 120)
    res = stepwiseSelect(x, y, criterion="bic", featureNames=["a", "b", "c"])
    assert set(res.termNames) == {"1", "a", "b*c", "c^2"}
    assert res.model.predict(x[:3]).shape == (3,)
    back = stepwiseSelect(x, y, criterion="loo", direction="backward")
    assert {"1", "x0", "x1*x2", "x2^2"} <= set(back.termNames)


def test_tuning():
    rng = np.random.default_rng(4)
    x = rng.uniform(-1, 1, (80, 2))
    y = x[:, 0] ** 2 + x[:, 1] + rng.normal(0, 0.02, 80)
    sel = tuneHyperparameters("linear", {"basis.degree": [1, 2, 3]}, x, y)
    assert sel.best.name == "basis.degree=2"


def test_bootstrapAgreesWithAnalyticBands():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 60)
    y = 1 + 2 * x - 3 * x ** 2 + rng.normal(0, 0.05, 60)
    m = createModel("quadratic").fit(x, y)
    b = bootstrap(m, x, y, xNew=[[0.3], [0.9]], nBoot=400, nJobs=4, seed=1)
    pi = m.predictInterval([[0.3], [0.9]], 0.95, "confidence")
    np.testing.assert_allclose(b.upper - b.lower, pi.upper - pi.lower, rtol=0.25)
    np.testing.assert_allclose(b.paramStd, m.result.stdErrors[:, 0], rtol=0.2)
    for method in ("wild", "pairs"):
        r = bootstrap(createModel("rbf-smooth"), x, y, xNew=[[0.5]], nBoot=30, method=method)
        assert r.lower[0, 0] < r.upper[0, 0] and r.nFailed == 0


def test_reportMarkdown():
    x = np.linspace(0, 1, 30)
    m = createModel("quadratic").fit(x, 1 + x ** 2)
    text = modelReport(m, title="T", diagnostics=diagnose(m))
    assert text.startswith("# T") and "## Parameters" in text and "## Residual diagnostics" in text


def test_handlerPhase4Commands(tmp_path):
    rh = RegressionHandler()
    rh.invoke("sampleBenchmark", problemName="branin", nSamples=35)
    rep = rh.invoke("autoFit", modelName="best", dataName="branin")
    assert rep["best"] and rep["modelName"] == "best"
    assert rh.invoke("diagnoseModel", modelName="best")["outputs"][0]["output"] == "branin"
    boot = rh.invoke("bootstrapModel", modelName="best", x=[[0.0, 5.0]], nBoot=20)
    assert boot["lower"][0][0] <= boot["upper"][0][0]
    path = str(tmp_path / "r.md")
    out = rh.invoke("exportReport", modelName="best", filePath=path)
    assert "## Model comparison" in out["markdown"] and open(path).read() == out["markdown"]
    assert rh.paramDict["documentResponse"]["titleDocument"].endswith("best")
    rh.invoke("setData", dataName="rs", x=np.random.default_rng(0).uniform(size=(40, 2)).tolist(),
              y=np.arange(40.0).tolist(), featureNames=["a", "b"])
    sw = rh.invoke("stepwiseSelect", modelName="sw", dataName="rs")
    assert all(t == "1" or t[0] in "ab" for t in sw["stepwise"]["selectedTerms"])
    assert rh.invoke("tuneModel", modelName="t", model="linear", grid={"basis.degree": [1, 2]},
                     dataName="rs")["modelName"] == "t"
    cmp = rh.invoke("compareModels", dataName="rs", candidates=["linear", '{"type": "idw"}'])
    assert len(cmp["ranking"]) == 2
    pts = rh.invoke("generateSamples", nSamples=6, xlimits=[[0, 1], [10, 20]], method="lhs")["x"]
    assert np.array(pts).shape == (6, 2)
    ids = {c["id"] for c in rh.buildCatalog()}
    assert {"autoFit", "compareModels", "tuneModel", "stepwiseSelect", "diagnoseModel", "bootstrapModel",
            "generateSamples", "sampleBenchmark", "exportReport"} <= ids
