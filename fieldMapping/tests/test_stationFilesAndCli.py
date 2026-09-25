"""Station CSV / manifest input and the command line (python -m pythonLibs.fieldMapping)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from pythonLibs.fieldMapping import (LayeredProfileMapper, loadMaterials, loadStations, planeSurface, readPvd,
                                     readVtu, structuredBox, writeStationTemplate, writeVtu)
from pythonLibs.fieldMapping.StationIO import readProfileCsv
from pythonLibs.fieldMapping.LayeredProfileMapper import Station

H0, H = 0.01, 0.03
DEPTH = np.linspace(0.0, H, 121)
TIMES = np.array([0.0, 10.0, 20.0])
MATS = {"1": {"density": 250.0, "cp": [[300.0, 1000.0], [2000.0, 1800.0]]}, "2": {"density": 2700.0, "cp": 900.0}}
POS = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]


def _temperature(a):
    return np.array([300.0 + a * 60.0 * t * np.exp(-DEPTH / 0.004) for t in TIMES])


def _writeCsv(path, header, rows):
    with open(path, "w") as f:
        f.write("# test data\n")
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.10g}" for v in r) + "\n")


@pytest.fixture
def case(tmp_path):
    mesh = structuredBox((0, 0, 0), (1, 1, H), (4, 4, 6))
    mesh.cellData["layer"] = np.where(mesh.cellCentroids()[:, 2] > H - H0, 1, 2).astype(np.int32)
    meshPath = str(tmp_path / "vol.vtu")
    writeVtu(meshPath, mesh)
    inline, entries = [], []
    for i, ((x, y), a) in enumerate(zip(POS, (1.0, 0.8, 1.2, 0.9))):
        T = _temperature(a)
        inline.append({"name": f"S{i}", "position": [x, y, H], "time": TIMES, "depth": DEPTH,
                       "temperature": T, "interfaces": [0.0, H0, H]})
        _writeCsv(tmp_path / f"S{i}.csv", ["time"] + [f"{d:.10g}" for d in DEPTH],
                  np.column_stack([TIMES, T]))
        entries.append({"name": f"S{i}", "position": [x, y, H], "interfaces": [0.0, H0, H], "file": f"S{i}.csv"})
    manifest = str(tmp_path / "stations.json")
    with open(manifest, "w") as f:
        json.dump({"stations": entries}, f)
    materials = str(tmp_path / "materials.json")
    with open(materials, "w") as f:
        json.dump(MATS, f)
    return {"mesh": mesh, "meshPath": meshPath, "inline": inline, "manifest": manifest, "materials": materials,
            "dir": tmp_path}


def test_csvStationsEqualInlineData(case):
    stations = loadStations(case["manifest"])
    for s, ref in zip(stations, case["inline"]):
        np.testing.assert_allclose(s["time"], ref["time"])
        np.testing.assert_allclose(s["depth"], ref["depth"])
        np.testing.assert_allclose(s["temperature"], ref["temperature"], rtol=1e-9)
    mats = loadMaterials(case["materials"])
    assert set(mats) == {1, 2}
    a = LayeredProfileMapper(case["mesh"], stations, mats).map()
    b = LayeredProfileMapper(case["mesh"], case["inline"], mats).map()
    np.testing.assert_allclose(a.temperature, b.temperature, rtol=1e-9)


def test_movingDepthGridAndInterfaceHistory(tmp_path):
    # recession: the depth grid and the interfaces change with time
    thick = np.array([0.030, 0.028, 0.025])
    depth = np.array([np.linspace(0.0, h, 5) for h in thick])
    temp = 300.0 + 100.0 * np.arange(15).reshape(3, 5)
    _writeCsv(tmp_path / "T.csv", ["time", "T1", "T2", "T3", "T4", "T5"], np.column_stack([TIMES, temp]))
    _writeCsv(tmp_path / "d.csv", ["time", "1", "2", "3", "4", "5"], np.column_stack([TIMES, depth]))
    _writeCsv(tmp_path / "b.csv", ["time", "b0", "b1", "b2"],
              np.column_stack([TIMES, np.zeros(3), thick - 0.02, thick]))
    with open(tmp_path / "m.json", "w") as f:
        json.dump([{"position": [0, 0, 0], "file": "T.csv", "depthFile": "d.csv", "interfacesFile": "b.csv"}], f)
    (s,) = loadStations(str(tmp_path / "m.json"))
    np.testing.assert_allclose(s["depth"], depth)
    np.testing.assert_allclose(s["interfaces"][:, 2], thick)
    assert s["name"] == "S1"
    st = Station.fromSpec(s)
    np.testing.assert_allclose(st.at(15.0)[2], [0.0, 0.0065, 0.0265])
    with pytest.raises(ValueError):
        readProfileCsv(str(tmp_path / "T.csv"))                   # header does not list depths


def test_templateFilesLoad(tmp_path):
    manifest = writeStationTemplate(str(tmp_path / "tpl"))
    (s,) = loadStations(manifest)
    assert np.shape(s["temperature"]) == (3, 11) and s["temperature"][2][0] == pytest.approx(1800.0)


def _cli(*args):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-m", "pythonLibs.fieldMapping", *args], capture_output=True, text=True,
                       env=env)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def test_cliLayeredEndToEnd(case):
    out = str(case["dir"] / "T.pvd")
    report = str(case["dir"] / "report.json")
    r = _cli("layered", "--mesh", case["meshPath"], "--stations", case["manifest"], "--materials",
             case["materials"], "--layers", "1", "2", "--out", out, "--report", report)
    assert r["summary"]["nTimes"] == 3 and r["summary"]["maxRelativeEnergyError"] < 1e-10
    assert json.load(open(report))["nStations"] == 4
    frames = readPvd(out)
    last = readVtu(frames[-1]["file"])
    assert last.pointData["temperature"].shape == (case["mesh"].nPoints,)
    assert last.cellData["temperature"].shape == (case["mesh"].nCells,)
    assert np.all(np.isfinite(last.cellData["temperature"]))


def test_cliFluxAndInfo(tmp_path):
    src = planeSurface(n=(12, 12), cellType="triangle", jitter=0.2)
    src.cellData["q"] = np.sin(5 * src.cellCentroids()[:, 0])
    tgt = structuredBox((0, 0, -0.03), (1, 1, 0), (5, 5, 2))
    writeVtu(str(tmp_path / "src.vtu"), src)
    writeVtu(str(tmp_path / "tgt.vtu"), tgt)
    info = _cli("info", str(tmp_path / "tgt.vtu"))
    assert info["nCells"] == 50 and info["measure"] == pytest.approx(0.03)
    r = _cli("flux", "--source", str(tmp_path / "src.vtu"), "--target", str(tmp_path / "tgt.vtu"), "--array", "q",
             "--point-flux", "--out", str(tmp_path / "mapped.vtu"))
    d = r["diagnostics"]
    assert abs(d["relativeErrorIn"]) < 1e-12 and abs(d["relativeErrorOut"]) < 1e-12
    assert "qNodalLoad" in readVtu(str(tmp_path / "mapped.vtu")).pointData


def test_cliReportsErrors(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-m", "pythonLibs.fieldMapping", "info", str(tmp_path / "missing.vtu")],
                       capture_output=True, text=True, cwd=str(tmp_path), env=env)
    assert p.returncode == 1 and p.stderr.startswith("error:")
