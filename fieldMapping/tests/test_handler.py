"""FieldMappingHandler as a toolBase service: files in, conservative mappings out."""
import json
import os

import numpy as np
import pytest

from pythonLibs.fieldMapping import FieldMappingHandler, planeSurface, readPvd, readVtu, structuredBox, writePvd, \
    writeVtu
from pythonLibs.tool.LabRegistry import listRegistered


@pytest.fixture
def files(tmp_path):
    src = planeSurface(n=(20, 20), cellType="triangle", jitter=0.2)
    src.cellData["q"] = np.sin(5 * src.cellCentroids()[:, 0]) + 0.2
    vol = structuredBox((0, 0, -0.03), (1, 1, 0), (4, 4, 3))
    vol.cellData["layer"] = np.where(vol.cellCentroids()[:, 2] > -0.01, 1, 2).astype(np.int32)
    paths = {"src": str(tmp_path / "src.vtu"), "vol": str(tmp_path / "vol.vtu")}
    writeVtu(paths["src"], src)
    writeVtu(paths["vol"], vol)
    # a 3-step source time series
    series = []
    for k, scale in enumerate((0.5, 1.0, 2.0)):
        s = planeSurface(n=(20, 20), cellType="triangle", jitter=0.2)
        s.cellData["q"] = scale * src.cellData["q"]
        f = str(tmp_path / f"src_{k}.vtu")
        writeVtu(f, s)
        series.append((float(k), f))
    paths["series"] = str(tmp_path / "src.pvd")
    writePvd(paths["series"], series)
    depth = np.linspace(0, 0.03, 301)
    stations = [{"position": [x, y, 0.0], "time": [0.0, 10.0], "depth": depth.tolist(), "interfaces": [0, 0.01, 0.03],
                 "temperature": [[300.0] * 301, (300 + 1200 * np.exp(-depth / 0.003)).tolist()]}
                for x, y in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]]
    paths["stations"] = str(tmp_path / "stations.json")
    with open(paths["stations"], "w") as f:
        json.dump(stations, f)
    paths["dir"] = str(tmp_path)
    return paths


def test_catalogAndRegistry():
    fm = FieldMappingHandler()
    ids = {c["id"] for c in fm.buildCatalog()}
    assert {"loadMesh", "mapSurfaceFlux", "mapLayeredProfiles", "extractSurface", "writeMesh"} <= ids
    assert listRegistered()["fieldMapping"] is FieldMappingHandler


def test_surfaceFluxThroughFiles(files):
    fm = FieldMappingHandler()
    fm.invoke("loadMesh", filePath=files["src"], meshName="flow")
    summary = fm.invoke("loadMesh", filePath=files["vol"], meshName="structure")
    assert summary["measure"] == pytest.approx(0.03)
    out = os.path.join(files["dir"], "mapped.vtu")
    r = fm.invoke("mapSurfaceFlux", sourceMesh="flow", targetMesh="structure", arrayName="q", pointFlux=True,
                  outputFile=out)
    d = r["diagnostics"]
    assert abs(d["relativeErrorIn"]) < 1e-13 and abs(d["relativeErrorOut"]) < 1e-13
    mapped = readVtu(out)
    assert {"q", "qHeatIn", "qHeatOut"} <= set(mapped.cellData) and "qNodalLoad" in mapped.pointData
    assert mapped.pointData["qNodalLoad"].sum() == pytest.approx(d["sourceHeatIn"] + d["sourceHeatOut"])


def test_surfaceFluxTimeSeries(files):
    fm = FieldMappingHandler()
    fm.invoke("loadMesh", filePath=files["src"], meshName="flow")
    fm.invoke("loadMesh", filePath=files["vol"], meshName="structure")
    out = os.path.join(files["dir"], "mapped.pvd")
    r = fm.invoke("mapSurfaceFlux", sourceMesh="flow", targetMesh="structure", arrayName="q",
                  sourceSeries=files["series"], outputFile=out)
    assert r["nFrames"] == 3
    heats = [s["targetHeatIn"] for s in r["diagnostics"]]
    np.testing.assert_allclose(np.array(heats) / heats[1], [0.5, 1.0, 2.0])
    assert [e["time"] for e in readPvd(out)] == [0.0, 1.0, 2.0]


def test_layeredProfilesThroughFiles(files):
    fm = FieldMappingHandler()
    fm.invoke("loadMesh", filePath=files["vol"], meshName="structure")
    out = os.path.join(files["dir"], "T.pvd")
    r = fm.invoke("mapLayeredProfiles", targetMesh="structure", stationsFile=files["stations"],
                  materials='{"1": {"density": 250, "cp": [[300, 1000], [2000, 1800]]}, "2": {"density": 2700, "cp": 900}}',
                  layers="[1, 2]", outputFile=out)
    assert r["summary"]["converged"] and r["summary"]["maxRelativeEnergyError"] < 1e-11
    assert r["nGroups"] == 8
    series = readPvd(out)
    assert len(series) == 2
    T = readVtu(series[-1]["file"]).pointData["temperature"]
    assert np.nanmax(T) <= 1500.0 + 1e-9 and np.nanmin(T) >= 300.0 - 1e-9


def test_extractSurfaceAndErrors(files):
    fm = FieldMappingHandler()
    fm.invoke("loadMesh", filePath=files["vol"], meshName="structure")
    b = fm.invoke("extractSurface", meshName="structure", surfaceName="skin")
    assert b["measure"] == pytest.approx(2 * (1 + 0.03 + 0.03))
    i = fm.invoke("extractSurface", meshName="structure", surfaceName="bond", kind="interface", layerArray="layer")
    assert i["measure"] == pytest.approx(1.0)
    with pytest.raises(Exception):
        fm.getMesh("missing")
