"""RegressionHandler as a toolBaseSecured service."""
import numpy as np
import pytest

from pythonLibs.regressionHandler import RegressionHandler
from pythonLibs.tool.LabRegistry import listRegistered


@pytest.fixture
def handler():
    rh = RegressionHandler()
    x = np.linspace(1, 5, 40)
    y = 28 + 4 * x - 12 / x ** 2 + np.random.default_rng(0).normal(0, 0.05, x.size)
    rh.invoke("setData", dataName="d", x=x.tolist(), y=y.tolist(), featureNames=["u"], outputNames=["v"])
    return rh


def test_catalogAndRegistry():
    rh = RegressionHandler()
    ids = {c["id"] for c in rh.buildCatalog()}
    assert {"setData", "fitModel", "predict", "crossValidateModel", "saveModel", "loadModel"} <= ids
    assert all(c["category"] == "regression" for c in rh.buildCatalog())
    assert listRegistered()["regression"] is RegressionHandler


def test_fitPredictWorkflow(handler):
    rep = handler.invoke("fitModel", modelName="m", model="poly3", dataName="d")
    assert rep["metrics"]["v"]["rSquared"] > 0.99
    assert len(rep["parameters"]["v"]) == 4
    out = handler.invoke("predict", modelName="m", x=[[2.0], [4.0]], level=0.9)
    assert np.all(np.array(out["lower"]) < np.array(out["mean"]))
    assert handler.invoke("predict", modelName="m", x=[3.0])["mean"][0][0] == pytest.approx(
        handler.getModel("m").predict([3.0])[0])
    assert handler.paramDict["executionResult"]["predict"]["result"]["modelName"] == "m"


def test_specFormsAndKwargAdaptation(handler):
    handler.invoke("fitModel", modelName="a", model={"type": "linearBasis", "basis": {"type": "polynomial", "degree": 2}},
                   data_name="d")
    handler.invoke("fitModel", modelName="b", model='{"type": "linearBasis", "solver": "robust"}', dataName="d")
    handler.invoke("fitModel", modelName="c", model="linear", dataName="d", options={"solver": "ridge"})
    names = [m["modelName"] for m in handler.invoke("listModels")["models"]]
    assert names == ["a", "b", "c"]


def test_crossValidationAndPersistence(handler, tmp_path):
    handler.invoke("fitModel", modelName="m", model="poly3", dataName="d")
    loo = handler.invoke("crossValidateModel", modelName="m", method="analytic")
    refit = handler.invoke("crossValidateModel", modelName="m", nFolds=40)
    assert loo["rmse"][0] == pytest.approx(refit["rmse"][0], rel=1e-10)
    path = str(tmp_path / "m.json")
    handler.invoke("saveModel", modelName="m", filePath=path)
    rep = handler.invoke("loadModel", filePath=path, modelName="m2")
    assert rep["equations"] == handler.invoke("getModelSummary", modelName="m")["equations"]
    assert path in handler.getExecutionContext()["output_files"]


def test_loadDataFromCsv(tmp_path):
    path = tmp_path / "d.csv"
    rows = ["a,b,y,w"] + [f"{i},{i * i},{2 * i + 1},1" for i in range(10)] + ["x,1,2,1"]
    path.write_text("\n".join(rows))
    rh = RegressionHandler()
    res = rh.invoke("loadDataFromFile", filePath=str(path), inputColumns=["a", "b"], outputColumns=["y"],
                    weightColumn="w")
    assert res["nSamples"] == 10 and res["droppedRows"] == 1
    with pytest.raises(KeyError):
        rh.invoke("loadDataFromFile", filePath=str(path), inputColumns=["zz"], outputColumns=["y"])


def test_errorsAreRecorded(handler):
    with pytest.raises(KeyError):
        handler.invoke("predict", modelName="missing", x=[1.0])
    err = handler.paramDict["executionResult"]["predict"]["error"]
    assert err["type"] == "KeyError"


# ---------------------------------------------------------------- extended commands
def test_inspectModelAspects():
    rh = RegressionHandler()
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (150, 2))
    y = rng.poisson(np.exp(0.3 + 0.8 * x[:, 0])).astype(float)
    rh.invoke("setData", dataName="c", x=x.tolist(), y=y.tolist())
    rep = rh.invoke("fitModel", modelName="g", model={"type": "glm", "family": "poisson"}, dataName="c")
    assert {"glmSummary", "residuals", "coefficients"} <= set(rep["aspects"])
    s = rh.invoke("inspectModel", modelName="g", aspect="glmSummary")["value"]
    assert s["family"] == "poisson(log)" and s["converged"]
    r = rh.invoke("inspectModel", modelName="g", aspect="residuals", aspectOptions={"kind": "pearson"})["value"]
    assert len(r) == 150
    with pytest.raises(ValueError):
        rh.invoke("inspectModel", modelName="g", aspect="termTable")
    with pytest.raises(ValueError):
        rh.invoke("inspectModel", modelName="g", aspect="residuals", aspectOptions={"evil": 1})
    with pytest.raises(ValueError):
        rh.invoke("inspectModel", modelName="g", aspect="__class__")


def test_jointPredictionCommands():
    rh = RegressionHandler()
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 3, (40, 2))
    rh.invoke("setData", dataName="s", x=x.tolist(), y=(np.sin(x[:, 0]) * np.cos(x[:, 1])).tolist())
    rh.invoke("fitModel", modelName="k", model={"type": "kriging", "corr": "matern52"}, dataName="s")
    pts = [[0.5, 0.5], [1.0, 2.0], [2.5, 1.5]]
    cov = rh.invoke("predictCovariance", modelName="k", x=pts)
    c = np.array(cov["covariance"][0])
    assert c.shape == (3, 3) and np.allclose(c, c.T)
    sims = np.array(rh.invoke("simulate", modelName="k", x=pts, nSamples=5, seed=1)["samples"])
    assert sims.shape == (5, 3, 1)
    blk = rh.invoke("predictBlock", modelName="k", blocks=[{"lower": [0, 0], "upper": [1, 1], "n": 4}, pts])
    assert blk["nPoints"] == [16, 3] and blk["variance"][1][0] <= np.max(np.diag(c)) + 1e-12
    k = rh.invoke("inspectModel", modelName="k", aspect="looDiagnostics")["value"]
    assert len(k["residuals"]) == 40
    ci = rh.invoke("inspectModel", modelName="k", aspect="parameterIntervals", aspectOptions={"level": 0.9})["value"]
    assert all(v["lower"] <= v["estimate"] <= v["upper"] for v in ci.values())


def test_gamTermsSensitivityAndQuantiles():
    rh = RegressionHandler()
    rh.invoke("sampleBenchmark", problemName="ishigami", nSamples=300, dataName="ish")
    rh.invoke("fitModel", modelName="pce", model="pce10", dataName="ish")
    sob = rh.invoke("sensitivityAnalysis", modelName="pce")
    assert sob["method"] == "pce"
    np.testing.assert_allclose(sob["first"], [0.3139, 0.4424, 0.0], atol=0.03)
    mc = rh.invoke("sensitivityAnalysis", modelName="pce", method="montecarlo", nSamples=1024, nBoot=20)
    assert mc["method"] == "montecarlo" and len(mc["firstInterval"]) == 3
    rh.invoke("fitModel", modelName="gam", model="gam", dataName="ish")
    terms = rh.invoke("predictTerms", modelName="gam", x=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])["terms"]
    assert set(terms) == {"s(x0)", "s(x1)", "s(x2)"} and len(terms["s(x0)"]["fit"]) == 2
    table = rh.invoke("inspectModel", modelName="gam", aspect="termTable")["value"]
    assert [row["term"] for row in table] == ["s(x0)", "s(x1)", "s(x2)"]
    qp = rh.invoke("quantileProcess", taus=[0.1, 0.5, 0.9], dataName="ish")
    coef = np.array(qp["coef"])
    assert coef.shape == (3, 4) and coef[0, 0] < coef[2, 0]


def test_variogramToKrigingAndMultiFidelity():
    rh = RegressionHandler()
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 5, (250, 2))
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    z = np.linalg.cholesky(np.exp(-d / 0.8) + 1e-10 * np.eye(250)) @ rng.standard_normal(250)
    rh.invoke("setData", dataName="field", x=x.tolist(), y=z.tolist())
    ev = rh.invoke("empiricalVariogram", dataName="field", maxDistance=2.5, variogramName="v")
    assert len(ev["gamma"]) > 5
    fit = rh.invoke("fitVariogram", variogramName="v", variogramModel="exponential", modelName="kv")
    assert 0.3 < fit["range"] < 2.0 and fit["kriging"]["type"] == "kriging"
    assert rh.invoke("predict", modelName="kv", x=[[1.0, 1.0]])["mean"][0][0] == pytest.approx(
        rh.getModel("kv").predict([[1.0, 1.0]]).ravel()[0])
    xl, xh = np.linspace(0, 1, 12), np.array([0.0, 0.4, 0.6, 1.0])
    f = lambda t: (6 * t - 2) ** 2 * np.sin(12 * t - 4)
    rh.invoke("setData", dataName="lo", x=xl.tolist(), y=(0.5 * f(xl) + 10 * (xl - 0.5) - 5).tolist())
    rh.invoke("setData", dataName="hi", x=xh.tolist(), y=f(xh).tolist())
    rep = rh.invoke("fitMultiFidelity", modelName="mf", dataNames=["lo", "hi"])
    assert rep["levels"] == ["lo", "hi"]
    t = np.linspace(0, 1, 30)
    pred = np.array(rh.invoke("predict", modelName="mf", x=np.column_stack([t, np.ones(30)]).tolist())["mean"])
    assert np.sqrt(np.mean((pred.ravel() - f(t)) ** 2)) < 1.0


def test_missingOutputsForCokriging():
    rh = RegressionHandler()
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 3, (30, 2))
    y = np.column_stack([np.sin(x[:, 0]), np.sin(x[:, 0]) + 0.3 * x[:, 1]]).tolist()
    for row in y[15:]:
        row[0] = None
    summ = rh.invoke("setData", dataName="h", x=x.tolist(), y=y)
    assert summ["outputs"][0]["missing"] == 15
    rep = rh.invoke("fitModel", modelName="ck", model={"type": "cokriging", "nStart": 1}, dataName="h")
    assert rep["type"] == "cokriging" and "coregionalization" in rep["aspects"]
    with pytest.raises(ValueError):
        rh.invoke("fitModel", modelName="lin", model="linear", dataName="h")
