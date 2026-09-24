"""Handler state consistency, text parameters, missing outputs and strict-JSON persistence."""
import json

import numpy as np
import pytest

from pythonLibs.regressionHandler import RegressionHandler
from pythonLibs.regressionHandler.core.SurrogateModelBase import SurrogateModelBase


@pytest.fixture
def rh():
    h = RegressionHandler()
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (60, 2))
    y = np.column_stack([np.sin(3 * x[:, 0]) + x[:, 1], np.cos(3 * x[:, 0])]) + 0.05 * rng.standard_normal((60, 2))
    h.setData(x=x.tolist(), y=y.tolist(), dataName="d", featureNames=["s", "t"], outputNames=["a", "b"])
    return h


def test_failedReportLeavesStateUnchanged(rh, monkeypatch):
    rh.fitModel("m", "linear", "d")
    before = rh.getModel("m")

    def broken(self):
        raise RuntimeError("summary failed")
    monkeypatch.setattr(SurrogateModelBase, "summary", broken)
    with pytest.raises(RuntimeError):
        rh.fitModel("m", "quadratic", "d")
    with pytest.raises(RuntimeError):
        rh.fitModel("new", "quadratic", "d")
    assert rh.getModel("m") is before and "new" not in rh._models and "new" not in rh._modelData


def test_selectionsAreClearedWhenAModelIsReplaced(rh):
    rh.compareModels("d", candidates=["linear", "quadratic"], modelName="m", nFolds=3)
    assert "m" in rh._selections
    rh.fitModel("m", "linear", "d")
    assert "m" not in rh._selections
    rh.compareModels("d", candidates=["linear", "quadratic"], modelName="m", nFolds=3)
    rh.deleteModel("m")
    assert "m" not in rh._selections


def test_crossValidationUsesTheModelsOwnData(rh):
    rh.fitModel("m", "linear", "d")
    expected = rh.crossValidateModel("m", nFolds=5)
    # the dataset is replaced after fitting: CV still refers to the data the model was fitted to
    rh.setData(x=[[0.0, 0.0], [1.0, 1.0], [0.5, 0.2], [0.3, 0.9]], y=[[1, 2], [3, 4], [5, 6], [7, 8]], dataName="d",
               featureNames=["s", "t"], outputNames=["a", "b"])
    assert rh.crossValidateModel("m", nFolds=5)["rmse"] == expected["rmse"]


def test_multiOutputAspectsAreReportedPerOutput(rh):
    rh.fitModel("g", "gam", "d")
    out = rh.inspectModel("g", "gamSummary")["value"]["outputs"]
    assert set(out) == {"a", "b"} and out["a"]["edf"] > 1
    assert set(rh.inspectModel("g", "termTable")["value"]["outputs"]) == {"a", "b"}
    assert "[b]" in rh.getModelSummary("g")["summary"]


def test_textParametersAreParsed(rh):
    rep = rh.invoke("fitModel", modelName="p", model='{"type": "linearBasis"}', dataName="d",
                    options='{"basis": {"type": "polynomial", "degree": 2}}')
    assert len(rep["parameters"]["a"]) == 6
    out = rh.invoke("predict", modelName="p", x="[[0.5, 0.5]]")
    assert np.array(out["mean"]).shape == (1, 2)
    rh.fitModel("g", "glm", "d")
    asp = rh.invoke("inspectModel", modelName="g", aspect="residuals", aspectOptions='{"kind": "pearson"}')
    assert len(asp["value"]["outputs"]["a"]) == 60


def test_variogramWorkflowUsesOneOutputAndItsObservedRows():
    h = RegressionHandler()
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (80, 3))
    z = np.sin(4 * x[:, 0]) * np.cos(3 * x[:, 1])
    y = np.column_stack([rng.standard_normal(80), z])
    y[::7, 1] = np.nan
    h.setData(x=x.tolist(), y=np.where(np.isnan(y), None, y).tolist(), dataName="d",
              featureNames=["east", "north", "time"], outputNames=["noise", "z"])
    ev = h.empiricalVariogram("d", output="z", columns=["east", "north"], variogramName="v", nBins=8)
    assert h._variograms["v"]["columns"] == [0, 1] and h._variograms["v"]["output"] == 1
    rep = h.fitVariogram("v", "gaussian", modelName="k")
    m = h.getModel("k")
    assert m.ny == 1 and m.outputNames == ["z"] and m.nTrain == int(np.sum(np.isfinite(y[:, 1])))
    assert rep["kriging"]["modelName"] == "k"


def test_quantileProcessAndMultiFidelitySkipMissingOutputs():
    h = RegressionHandler()
    x = np.linspace(0, 1, 40)
    y = np.column_stack([2 + x, 1 + 3 * x])
    y[5, 1] = np.nan
    h.setData(x=x.tolist(), y=np.where(np.isnan(y), None, y).tolist(), dataName="d")
    res = h.quantileProcess(taus="[0.25, 0.5]", dataName="d", output=1)
    assert res["nUsed"] == 39
    np.testing.assert_allclose(res["coef"], [[1, 3], [1, 3]], atol=1e-8)
    xl = np.linspace(0, 1, 12)
    yl = np.column_stack([np.sin(6 * xl), xl])
    yl[3, 0] = np.nan
    h.setData(x=xl.tolist(), y=np.where(np.isnan(yl), None, yl).tolist(), dataName="lo")
    xh = np.linspace(0, 1, 6)
    h.setData(x=xh.tolist(), y=np.column_stack([1.5 * np.sin(6 * xh) + 0.2, xh]).tolist(), dataName="hi")
    rep = h.fitMultiFidelity("mf", '["lo", "hi"]')
    assert rep["levels"] == ["lo", "hi"] and np.all(np.isfinite(h.getModel("mf").predict([[0.35, 1.0]])))


def test_predictBlockValidation(rh):
    rh.fitModel("k", {"type": "kriging", "poly": "constant"}, "d")
    box = {"lower": [0.2, 0.2], "upper": [0.4, 0.4], "n": 3}
    ok = rh.predictBlock("k", [box], nPerDim=2)
    assert ok["nPoints"] == [9]
    for blocks, weights in (([], None), ([box], [[1.0] * 9, [1.0] * 9]), ([box], [[-1.0] + [1.0] * 8]),
                            ([box], [[0.0] * 9]), ([box], [[np.nan] + [1.0] * 8]), ([box], [[1.0] * 4])):
        with pytest.raises(ValueError):
            rh.predictBlock("k", blocks, weights)
    with pytest.raises(ValueError):
        rh.predictBlock("k", [{"lower": [0, 0], "upper": [1, 1], "n": 0}])


def test_savedFilesAreStrictJsonAndKeepTypes(rh, tmp_path):
    from pythonLibs.regressionHandler.core.Serialization import compactArrays, expandArrays
    rh.fitModel("m", "linear", "d")
    m = rh.getModel("m")
    import dataclasses
    m.metrics[0] = dataclasses.replace(m.metrics[0], maxAbsError=float("inf"))
    m.metrics[1] = dataclasses.replace(m.metrics[1], mae=float("nan"))
    path = tmp_path / "m.json"
    for compact in (False, True):
        m.save(str(path), compact=compact)

        def reject(token):
            raise ValueError(f"non-standard token {token}")
        json.loads(path.read_text(), parse_constant=reject)
        m2 = SurrogateModelBase.load(str(path))
        assert m2.metrics[0].maxAbsError == float("inf") and np.isnan(m2.metrics[1].mae)
        np.testing.assert_allclose(m2.predict([[0.3, 0.4]]), m.predict([[0.3, 0.4]]), rtol=1e-15)
    data = {"ints": [1, None, 3] * 30, "bools": [True, None] * 40, "big": [2 ** 70] * 70, "mixed": [True, 2] * 40}
    back = expandArrays(json.loads(json.dumps(compactArrays(data))))
    assert back == data
    assert all(type(a) is type(b) for a, b in zip(back["ints"] + back["mixed"], data["ints"] + data["mixed"]))
