"""Layered 1-D profile mapping: exact reproduction, energy conservation, bounds, layer normalization."""
import numpy as np
import pytest

from pythonLibs.fieldMapping.LayeredProfileMapper import LayeredProfileMapper, Station
from pythonLibs.fieldMapping.Material import Material
from pythonLibs.fieldMapping.io.Vtu import readPvd, readVtu
from pythonLibs.fieldMapping.mesh.Generators import sphericalShell, structuredBox

H0, H = 0.01, 0.03
DEPTH = np.linspace(0.0, H, 601)
MATS = {1: {"density": 250.0, "cp": [[300.0, 1000.0], [2000.0, 1800.0]]}, 2: {"density": 2700.0, "cp": 900.0}}


def _slab(nz=6, cellType="hexahedron", top=H0, n=4):
    m = structuredBox((0, 0, 0), (1, 1, H), (n, n, nz), cellType)
    m.cellData["layer"] = np.where(m.cellCentroids()[:, 2] > H - top, 1, 2).astype(np.int32)
    return m


def _stations(profile, times=(0.0, 10.0), interfaces=(0.0, H0, H), amps=(1.0, 1.0, 1.0, 1.0)):
    pos = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    return [{"position": [x, y, H], "time": list(times), "depth": DEPTH, "interfaces": list(interfaces),
             "temperature": np.array([profile(t, a) for t in times]), "name": f"s{i}"}
            for i, ((x, y), a) in enumerate(zip(pos, amps))]


def _steep(t, a):
    return 300.0 + a * 1500.0 * np.exp(-DEPTH / (0.001 + 0.0004 * t)) * (t > 0)


# ---------------------------------------------------------------- building blocks
def test_materialEnergyIsExact():
    mat = Material(250.0, [[300.0, 1000.0], [800.0, 1500.0], [2000.0, 1800.0]])
    t = np.linspace(100.0, 2600.0, 40)
    fine = np.linspace(300.0, 1.0, 2)                                        # unused placeholder
    del fine
    for T in t:
        grid = np.linspace(300.0, T, 20001)
        ref = np.trapezoid(mat.cp(grid), grid)
        assert mat.energy(T) == pytest.approx(ref, rel=1e-7, abs=1e-6)
    h = 1e-4
    np.testing.assert_allclose((mat.energy(t + h) - mat.energy(t - h)) / (2 * h), mat.cp(t), rtol=1e-6)


def test_stationTimeInterpolation():
    s = Station([0, 0, 0], [0.0, 2.0], [0.0, 1.0], [[300.0, 300.0], [500.0, 400.0]], [0.0, 1.0])
    d, T, b = s.at(0.5)
    np.testing.assert_allclose(T, [350.0, 325.0])
    np.testing.assert_allclose(s.at(5.0)[1], [500.0, 400.0])


# ---------------------------------------------------------------- reproduction
@pytest.mark.parametrize("cellType", ["hexahedron", "tetra", "wedge", "quadraticHexahedron", "polyhedron"])
def test_linearProfileIsReproducedExactly(cellType):
    # linear in depth within each layer (kink at the interface) and constant cp: nodal values are exact
    prof = lambda t, a: np.where(DEPTH < H0, 1000.0 - 40000.0 * DEPTH, 600.0 - 10000.0 * (DEPTH - H0))   # noqa: E731
    mats = {1: {"density": 250.0, "cp": 1200.0}, 2: {"density": 2700.0, "cp": 900.0}}
    m = _slab(nz=6, cellType=cellType)
    res = LayeredProfileMapper(m, _stations(prof), mats, layers=[1, 2]).map()
    z = H - m.points[:, 2]
    exact = np.where(z < H0, 1000.0 - 40000.0 * z, 600.0 - 10000.0 * (z - H0))
    np.testing.assert_allclose(res.temperature[-1], exact, atol=1e-8)
    assert res.summary()["maxRelativeNodalEnergyError"] < 1e-12


def test_layerNormalizationMatchesInterfaces():
    # 3-D top layer 0.012 thick, station top layer 0.010: interface nodes take the station interface value
    prof = lambda t, a: np.where(DEPTH < H0, 1000.0 - 40000.0 * DEPTH, 600.0 - 10000.0 * (DEPTH - H0))   # noqa: E731
    mats = {1: {"density": 250.0, "cp": 1200.0}, 2: {"density": 2700.0, "cp": 900.0}}
    m = structuredBox((0, 0, 0), (1, 1, H), (3, 3, 10))                     # nodes at z = 0.018 = H - 0.012
    m.cellData["layer"] = np.where(m.cellCentroids()[:, 2] > H - 0.012, 1, 2).astype(np.int32)
    res = LayeredProfileMapper(m, _stations(prof), mats, layers=[1, 2]).map()
    T = res.temperature[-1]
    at = np.isclose(m.points[:, 2], H - 0.012)
    np.testing.assert_allclose(T[at], 600.0, atol=1e-8)
    np.testing.assert_allclose(T[np.isclose(m.points[:, 2], H)], 1000.0, atol=1e-8)
    # a profile linear through the whole depth: layer-normalized puts the station interface value (z = 0.010)
    # on the 3-D interface, depth mode the value at the 3-D depth (0.012); both are exact nodal fields
    lin = lambda t, a: 1000.0 - 20000.0 * DEPTH                                          # noqa: E731
    r1 = LayeredProfileMapper(m, _stations(lin), mats, layers=[1, 2]).map()
    r2 = LayeredProfileMapper(m, _stations(lin), mats, layers=[1, 2], depthMode="depth").map()
    np.testing.assert_allclose(r1.temperature[-1][at], 800.0, atol=1e-8)
    np.testing.assert_allclose(r2.temperature[-1][at], 760.0, atol=1e-8)


# ---------------------------------------------------------------- conservation
@pytest.mark.parametrize("cellType", ["hexahedron", "tetra", "quadraticHexahedron"])
def test_energyIsConservedForSteepProfiles(cellType):
    m = _slab(nz=4, cellType=cellType)
    stations = _stations(_steep, times=(0.0, 5.0, 20.0), amps=(1.0, 0.8, 1.2, 0.9))
    mp = LayeredProfileMapper(m, stations, MATS, layers=[1, 2])
    res = mp.map()
    s = res.summary()
    assert s["converged"] and s["maxRelativeEnergyError"] < 1e-11
    assert s["maxRelativeNodalEnergyError"] > 1e-2                           # the correction is needed
    assert mp.nGroups == 8
    # no new extrema: within the station range
    T = res.temperature[-1]
    allT = np.concatenate([st["temperature"][-1] for st in stations])
    assert T.min() >= allT.min() - 1e-9 and T.max() <= allT.max() + 1e-9


def test_referenceEnergyMatchesTheOneDimensionalIntegral():
    # identical stations and matching layer thicknesses: reference energy = area x int rho e(T(z)) dz
    m = _slab(nz=3, n=2)
    stations = _stations(_steep, times=(20.0,))
    mp = LayeredProfileMapper(m, stations, MATS, layers=[1, 2], conservation="layer", referenceOrder=16)
    d = mp.map().diagnostics[0]
    T = _steep(20.0, 1.0)
    zf = np.linspace(0, H, 200001)
    Tf = np.interp(zf, DEPTH, T)
    mats = [Material.fromSpec(MATS[1]), Material.fromSpec(MATS[2])]
    exact = [np.trapezoid(np.where(zf <= H0, mats[0].volumetricEnergy(Tf), 0.0), zf),
             np.trapezoid(np.where(zf >= H0, mats[1].volumetricEnergy(Tf), 0.0), zf)]
    np.testing.assert_allclose(d["referenceEnergy"], exact, rtol=2e-4)
    np.testing.assert_allclose(d["finalEnergy"], d["referenceEnergy"], rtol=1e-12)


def test_curvedMultiLayerShellWithAllWeightings():
    sh = sphericalShell((0.95, 0.985, 1.0), n=4, layers=(2, 2), quadratic=True)
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(8, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    dz = np.linspace(0, 0.06, 400)
    st = [{"position": d, "time": [0.0, 5.0], "depth": dz, "interfaces": [0.0, 0.012, 0.05],
           "temperature": np.array([300 + 0 * dz, 300 + 1400 * (1 + 0.3 * d[2]) * np.exp(-dz / 0.004)])} for d in dirs]
    mats = {0: {"density": 300.0, "cp": [[300, 1100], [2500, 2000]]}, 1: {"density": 1600.0, "cp": 1000.0}}
    for w in ("idw", "kriging", "nearest"):
        mp = LayeredProfileMapper(sh, st, mats, weights=w)
        s = mp.map().summary()
        assert s["converged"] and s["maxRelativeEnergyError"] < 1e-11
    # station weights: partition of unity, exact at the stations
    W = mp._weights(mp.stationFoot)
    np.testing.assert_allclose(W.toDense(), np.eye(8), atol=1e-12)
    mp2 = LayeredProfileMapper(sh, st, mats, weights="kriging")
    rows = mp2._nodeWeights.toDense().sum(axis=1)
    np.testing.assert_allclose(rows, 1.0, atol=1e-10)


def test_conservationGroupsAndSeriesOutput(tmp_path):
    m = _slab(nz=4)
    stations = _stations(_steep, times=(0.0, 10.0))
    for mode, groups in (("global", 1), ("layer", 2), ("stationLayer", 8)):
        mp = LayeredProfileMapper(m, stations, MATS, layers=[1, 2], conservation=mode)
        assert mp.nGroups == groups
    res = mp.map(times=[5.0, 10.0])
    files = mp.writeSeries(res, str(tmp_path / "temperature.pvd"))
    assert len(files) == 2
    back = readPvd(str(tmp_path / "temperature.pvd"))
    np.testing.assert_allclose(readVtu(back[1]["file"]).pointData["temperature"], res.temperature[1])
