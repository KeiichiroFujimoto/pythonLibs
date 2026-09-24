"""Conservative surface flux transfer: exact conservation per sign, reproduction and convergence."""
import numpy as np
import pytest

from pythonLibs.fieldMapping.SurfaceFluxMapper import SurfaceFluxMapper
from pythonLibs.fieldMapping.mesh.Generators import planeSurface, sphereSurface, sphericalShell

TOL = 1e-13


def _field(x):
    return np.sin(6 * x[:, 0]) + 0.3 * x[:, 1]                       # changes sign


def _cellAverages(mesh, f, order=8):
    q = mesh.quadrature(order)
    area = np.bincount(q.cell, weights=q.weight, minlength=mesh.nCells)
    return np.bincount(q.cell, weights=q.weight * f(q.x), minlength=mesh.nCells) / np.where(area > 0, area, 1)


def _check(res):
    d = res.diagnostics
    assert abs(d["relativeErrorIn"]) < TOL and abs(d["relativeErrorOut"]) < TOL
    assert np.all(res.cellHeatPlus >= 0) and np.all(res.cellHeatMinus <= 0)
    total = d["sourceHeatIn"] + d["sourceHeatOut"] - d["droppedHeatIn"] - d["droppedHeatOut"]
    assert d["nodalLoadTotal"] == pytest.approx(total, rel=1e-12, abs=1e-15)


def test_identicalMeshesReproduceTheSource():
    tgt = planeSurface(n=(6, 5), cellType="quad", jitter=0.25, seed=2)
    lin = 2 + tgt.points[:, 0] - 3 * tgt.points[:, 1]
    res = SurfaceFluxMapper(tgt, tgt).map(lin, "point")
    exact = tgt.quadrature(4)
    ci = exact.cellIntegrals(exact.interp @ lin, tgt.nCells)
    np.testing.assert_allclose(res.cellHeatPlus + res.cellHeatMinus, ci, atol=1e-15)
    q = np.random.default_rng(0).normal(size=tgt.nCells)
    res2 = SurfaceFluxMapper(tgt, tgt).map(q, "cell")
    np.testing.assert_allclose(res2.cellFlux, q, atol=1e-14)
    _check(res2)


@pytest.mark.parametrize("srcType,tgtType", [("triangle", "quad"), ("quad", "quadraticTriangle"),
                                             ("polygon", "quadraticQuad"), ("quadraticTriangle", "triangle")])
def test_nonMatchingPlanarMeshesConservePerSign(srcType, tgtType):
    src = planeSurface(n=(23, 17), cellType=srcType, jitter=0.3, seed=0)
    tgt = planeSurface(n=(7, 9), cellType=tgtType, jitter=0.2, seed=1)
    m = SurfaceFluxMapper(src, tgt)
    for loc, vals in (("cell", _field(src.cellCentroids())), ("point", _field(src.points))):
        res = m.map(vals, loc, pointFlux=True)
        _check(res)
        d = res.diagnostics
        assert d["pointFluxHeat"] == pytest.approx(d["sourceHeatIn"] + d["sourceHeatOut"], rel=1e-12)
        assert d["droppedHeatIn"] == 0 and d["orphanTriangles"] == 0


def test_pointFluxRespectsGroupsAndSigns():
    src = planeSurface(n=(30, 30), cellType="triangle", jitter=0.3)
    tgt = planeSurface(n=(8, 8), cellType="quad", jitter=0.2, seed=3)
    m = SurfaceFluxMapper(src, tgt)
    positive = np.abs(_field(src.points)) + 0.1
    groups = (tgt.cellCentroids()[:, 0] > 0.5).astype(int)
    res = m.map(positive, "point", pointFlux=True, groups=groups)
    assert res.pointFlux.min() >= 0.0                                  # incoming heat only: no negative nodes
    for g in (0, 1):
        c = m._massRows(np.where(groups == g, 0, 1))
        assert (c @ res.pointFlux)[0] == pytest.approx(res.cellHeatPlus[groups == g].sum(), rel=1e-12)


def test_curvedSurfaceWithInwardSourceNormals():
    src = sphereSurface(1.0, 24, "triangle")
    src.connectivity = src.connectivity.reshape(-1, 3)[:, ::-1].ravel()          # inward, as some flow solvers write
    shell = sphericalShell((0.8, 1.0), n=6, layers=(1,), quadratic=True)
    m = SurfaceFluxMapper(src, shell)
    assert m.orientationSign == -1.0
    f = lambda x: x[:, 2] + 0.3                                                   # noqa: E731
    res = m.map(f(src.points), "point")
    _check(res)
    inner = np.linalg.norm(m.target.cellCentroids(), axis=1) < 0.9
    assert np.all(res.cellHeatPlus[inner] == 0) and np.all(res.cellHeatMinus[inner] == 0)


def test_convergenceUnderJointRefinement():
    f = lambda x: np.cos(3 * x[:, 0]) * np.sin(2 * x[:, 1]) + 0.2              # noqa: E731
    errs = []
    for k in (1, 2, 4):
        src = planeSurface(n=(12 * k, 12 * k), cellType="triangle", jitter=0.25, seed=k)
        tgt = planeSurface(n=(5 * k, 4 * k), cellType="quad", jitter=0.2, seed=10 + k)
        res = SurfaceFluxMapper(src, tgt).map(f(src.points), "point")
        exact = _cellAverages(tgt, f)
        area = tgt.cellMeasures()
        errs.append(np.sqrt(np.sum(area * (res.cellFlux - exact) ** 2)))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.7), rates                                           # second order for smooth fields


def test_partialTargetReportsDroppedHeat():
    src = planeSurface(n=(20, 20), cellType="triangle")
    tgt = planeSurface(n=(10, 10), cellType="quad")
    half = tgt.cellCentroids()[:, 0] < 0.5
    q = np.abs(_field(src.points)) + 0.1
    for mode in ("nearest", "drop"):
        m = SurfaceFluxMapper(src, tgt, targetCells=half, unmatched=mode, maxDistance=0.02)
        res = m.map(q, "point")
        _check(res)
        d = res.diagnostics
        assert d["droppedHeatIn"] > 0.3 * d["sourceHeatIn"]
        assert np.all(res.cellHeatPlus[~half] == 0)


def test_linearReconstructionIsConservativeAndSharper():
    f = lambda x: np.exp(-((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2) / 0.05) - 0.2    # noqa: E731
    coarse = planeSurface(n=(10, 10), cellType="quad")
    fine = planeSurface(n=(40, 40), cellType="triangle", jitter=0.2)
    m = SurfaceFluxMapper(coarse, fine)
    qc = _cellAverages(coarse, f)
    exact = _cellAverages(fine, f)
    area = fine.cellMeasures()
    err = {}
    for rec in ("constant", "linear"):
        res = m.map(qc, "cell", reconstruction=rec)
        _check(res)
        err[rec] = np.sqrt(np.sum(area * (res.cellFlux - exact) ** 2))
    assert err["linear"] < 0.6 * err["constant"]                                 # limited near peak and zero line


def test_repeatedMappingReusesTheGeometry():
    src = planeSurface(n=(15, 15), cellType="triangle", jitter=0.2)
    tgt = planeSurface(n=(6, 6), cellType="quad")
    m = SurfaceFluxMapper(src, tgt)
    a = m.map(_field(src.points), "point")
    b = m.map(2.0 * _field(src.points), "point")
    np.testing.assert_allclose(b.cellFlux, 2.0 * a.cellFlux, rtol=1e-12, atol=1e-15)
