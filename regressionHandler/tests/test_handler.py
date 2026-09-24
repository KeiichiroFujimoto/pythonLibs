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
