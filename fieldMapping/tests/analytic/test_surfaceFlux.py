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
        c = m._nodalRows(np.where(groups == g, 0, 1))
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
    assert np.all(rates > 1.7), rates                                           # second order for a smooth field


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


# ---------------------------------------------------------------- face-area weighting
def _rectGrid(xs, ys, z=None):
    from pythonLibs.fieldMapping.mesh.Mesh import UnstructuredMesh
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size) if z is None else z(X.ravel(), Y.ravel())])
    ny = len(ys)
    idx = lambda i, j: i * ny + j                                                 # noqa: E731
    quads = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
             for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
    return UnstructuredMesh.fromBlocks(pts, [(9, np.array(quads))])


def test_gradedGridsAreAreaWeightedExactly():
    # strongly graded source (face areas vary ~100x) onto a uniform target: target heat = sum q_s |s cap t|
    xs = np.concatenate([[0.0], np.cumsum(np.geomspace(0.002, 0.2, 14))])
    xs = xs / xs[-1]
    ys = np.linspace(0, 1, 9) ** 1.7
    xt, yt = np.linspace(0, 1, 6), np.linspace(0, 1, 5)
    src, tgt = _rectGrid(xs, ys), _rectGrid(xt, yt)
    q = np.random.default_rng(3).normal(size=src.nCells)
    res = SurfaceFluxMapper(src, tgt).map(q, "cell")

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))
    ref = np.zeros(tgt.nCells)
    k = 0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            t = 0
            for a in range(len(xt) - 1):
                for b in range(len(yt) - 1):
                    ref[t] += q[k] * overlap(xs[i], xs[i + 1], xt[a], xt[a + 1]) * \
                        overlap(ys[j], ys[j + 1], yt[b], yt[b + 1])
                    t += 1
            k += 1
    np.testing.assert_allclose(res.cellHeatPlus + res.cellHeatMinus, ref, atol=1e-15)
    np.testing.assert_allclose(res.cellFlux * tgt.cellMeasures(), ref, atol=1e-15)


def test_warpedFacesUseTheChosenFaceArea():
    from pythonLibs.fieldMapping.SurfaceFluxMapper import faceAreas
    warp = lambda x, y: 0.08 * np.sin(7 * x) * np.cos(5 * y)                  # noqa: E731
    src = _rectGrid(np.linspace(0, 1, 13), np.linspace(0, 1, 11), warp)
    tgt = _rectGrid(np.linspace(0, 1, 6), np.linspace(0, 1, 5), warp)
    q = 1.0 + 0.5 * src.cellCentroids()[:, 0]
    vec, fan, fe = (faceAreas(src, c) for c in ("vector", "fan", "fe"))
    assert np.all(vec <= fan + 1e-15) and np.max(fan - vec) > 1e-6          # warped: conventions differ
    for conv, areas in (("vector", vec), ("fan", fan), ("fe", fe)):
        res = SurfaceFluxMapper(src, tgt, sourceArea=conv).map(q, "cell")
        assert res.diagnostics["sourceHeatIn"] == pytest.approx(np.sum(q * areas), rel=1e-14)
        assert res.cellHeatPlus.sum() == pytest.approx(np.sum(q * areas), rel=1e-14)
    # the solver's own face areas, given as a cell array
    src.cellData["faceArea"] = vec * 1.01
    res = SurfaceFluxMapper(src, tgt, sourceArea="faceArea").map(q, "cell")
    assert res.cellHeatPlus.sum() == pytest.approx(np.sum(q * vec * 1.01), rel=1e-14)
    # target flux times the target face area (any convention) is the target face heat
    for conv in ("vector", "fe"):
        m = SurfaceFluxMapper(src, tgt, targetArea=conv)
        r = m.map(q, "cell")
        np.testing.assert_allclose(r.cellFlux * faceAreas(tgt, conv), r.cellHeatPlus, rtol=1e-13)
        assert r.nodalLoads.sum() == pytest.approx(r.cellHeatPlus.sum(), rel=1e-13)


def test_nodalFluxIsFaceAreaWeighted():
    xs = np.concatenate([[0.0], np.cumsum(np.geomspace(0.01, 0.3, 8))])
    tgt = _rectGrid(xs / xs[-1], np.linspace(0, 1, 5))
    src = planeSurface(n=(30, 30), cellType="triangle", jitter=0.2)
    m = SurfaceFluxMapper(src, tgt)
    # uniform flux: every nodal value equals it (area-weighted average of equal values, no correction)
    res = m.map(np.full(src.nCells, 3.0), "cell", pointFlux=True)
    used = np.unique(tgt.connectivity)
    np.testing.assert_allclose(res.pointFlux[used], 3.0, rtol=1e-13)
    # general flux: sum_i a_i q_i = heat with tributary areas a_i = sum_f A_f / 4
    q = np.abs(_field(src.cellCentroids())) + 0.1
    res = m.map(q, "cell", pointFlux=True)
    area = tgt.cellMeasures()
    a = np.zeros(tgt.nPoints)
    np.add.at(a, tgt.connectivity.reshape(-1, 4), np.repeat(area[:, None] / 4, 4, axis=1))
    assert a @ res.pointFlux == pytest.approx(res.diagnostics["sourceHeatIn"], rel=1e-12)
    assert res.pointFlux.min() >= 0.0


def test_clip_moves_only_the_covered_part_of_a_larger_source():
    """A 1 m x 1 m patch of a uniformly heated 2 m x 2 m plate whose triangles straddle the patch
    edges: "clip" gives exactly q0 on every patch facet and q0 x 1 m² of heat, the rest dropped;
    "renormalize" (for targets that cover the source) piles straddling heat onto the edges."""
    import numpy as np

    from pythonLibs.fieldMapping import SurfaceFluxMapper
    from pythonLibs.fieldMapping.mesh.Generators import planeSurface

    q0 = 2.0e5
    source = planeSurface((0.0, 0.0), (2.0, 2.0), (3, 3), cellType="triangle")          # coarse, 0.67 m cells
    patch = planeSurface((0.45, 0.55), (1.45, 1.55), (8, 8), cellType="triangle")       # edges inside cells
    mapper = SurfaceFluxMapper(source, patch, unmatched="drop")
    values = np.full(source.nCells, q0)
    clip = mapper.map(values, "cell", sourceCoverage="clip")
    np.testing.assert_allclose(clip.cellFlux, q0, rtol=1e-9)
    np.testing.assert_allclose(clip.diagnostics["targetHeatIn"], q0 * 1.0, rtol=1e-9)    # q0 x 1 m²
    assert abs(clip.diagnostics["relativeErrorIn"]) < 1e-12                           # moved + dropped = source
    renorm = mapper.map(values, "cell")
    assert renorm.cellFlux.max() > 1.5 * q0                                           # edge pile-up


# ---------------------------------------------------------------- reverse (intensive) transfer
def test_mapBackReproducesIdenticalMeshesAndConstants():
    mesh = planeSurface(n=(6, 5), cellType="quad", jitter=0.25, seed=2)
    T = np.random.default_rng(1).uniform(300.0, 1900.0, mesh.nCells)
    back = SurfaceFluxMapper(mesh, mesh).mapBack(T)
    np.testing.assert_allclose(back.cellValues, T, rtol=1e-13)
    src = planeSurface(n=(23, 17), cellType="triangle", jitter=0.3, seed=0)
    tgt = planeSurface(n=(7, 9), cellType="quadraticQuad", jitter=0.2, seed=1)
    res = SurfaceFluxMapper(src, tgt).mapBack(np.full(tgt.nPoints, 1234.5), "point")
    np.testing.assert_allclose(res.cellValues, 1234.5, rtol=1e-13)
    np.testing.assert_allclose(res.pointValues, 1234.5, rtol=1e-13)


def test_mapBackConservesTheOverlapIntegralWithoutNewExtrema():
    src = planeSurface(n=(9, 7), cellType="triangle", jitter=0.3, seed=4)
    tgt = planeSurface(n=(31, 29), cellType="quad", jitter=0.2, seed=5)
    T = 300.0 + 1500.0 * np.abs(_field(tgt.cellCentroids()))
    res = SurfaceFluxMapper(src, tgt).mapBack(T)
    d = res.diagnostics
    assert abs(d["relativeIntegralError"]) < 1e-13
    assert d["overlapIntegral"] == pytest.approx(np.sum(tgt.cellMeasures() * T), rel=1e-12)   # full cover
    assert res.cellValues.min() >= T.min() - 1e-9 and res.cellValues.max() <= T.max() + 1e-9
    assert res.pointValues.min() >= T.min() - 1e-9 and res.pointValues.max() <= T.max() + 1e-9


def test_mapBackConvergesToTheSourceCellAverage():
    f = lambda x: 300.0 + 800.0 * (1.0 + np.cos(3 * x[:, 0]) * np.sin(2 * x[:, 1]))   # noqa: E731
    src = planeSurface(n=(6, 6), cellType="triangle", jitter=0.2, seed=7)
    exact = _cellAverages(src, f)
    errs = []
    for k in (1, 2, 4):
        tgt = planeSurface(n=(10 * k, 11 * k), cellType="quad", jitter=0.2, seed=20 + k)
        res = SurfaceFluxMapper(src, tgt).mapBack(_cellAverages(tgt, f))
        errs.append(np.max(np.abs(res.cellValues - exact)))
    assert errs[-1] < 0.3 * errs[0], errs


def test_mapBackOnAPatchLeavesUncoveredSourceUnset():
    source = planeSurface((0.0, 0.0), (2.0, 2.0), (6, 6), cellType="triangle")
    patch = planeSurface((0.45, 0.55), (1.45, 1.55), (8, 8), cellType="triangle")
    res = SurfaceFluxMapper(source, patch, unmatched="drop").mapBack(np.full(patch.nCells, 900.0))
    covered = res.cellCoverage > 0
    assert 0 < covered.sum() < source.nCells
    np.testing.assert_allclose(res.cellValues[covered], 900.0, rtol=1e-13)
    assert np.all(np.isnan(res.cellValues[~covered]))
    assert res.diagnostics["overlapArea"] == pytest.approx(1.0, rel=1e-9)            # the 1 m² patch
    partial = covered & (res.cellCoverage < 0.999)
    assert partial.any()                                                              # straddling cells


def test_mapBackProjectionReproducesLinearFieldsExactly():
    lin = lambda x: 700.0 + 300.0 * x[:, 0] - 120.0 * x[:, 1]                    # noqa: E731
    for srcType in ("triangle", "quad"):
        src = planeSurface(n=(7, 6), cellType=srcType, jitter=0.25, seed=3)
        tgt = planeSurface(n=(19, 23), cellType="triangle", jitter=0.2, seed=4)
        res = SurfaceFluxMapper(src, tgt).mapBack(lin(tgt.points), "point", nodal="projection")
        np.testing.assert_allclose(res.pointValues, lin(src.points), rtol=1e-10)
        avg = SurfaceFluxMapper(src, tgt).mapBack(lin(tgt.points), "point")
        assert np.max(np.abs(avg.pointValues - lin(src.points))) > 1.0              # averaging smears edges


def test_mapBackProjectionIsSharperOnAPeak():
    f = lambda x: 300.0 + 1500.0 * np.exp(-((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.45) ** 2) / 0.04)   # noqa: E731
    src = planeSurface(n=(10, 10), cellType="triangle", jitter=0.2, seed=8)
    tgt = planeSurface(n=(60, 60), cellType="quad", jitter=0.2, seed=9)
    m = SurfaceFluxMapper(src, tgt)
    err = {mode: np.max(np.abs(m.mapBack(f(tgt.points), "point", nodal=mode).pointValues - f(src.points)))
           for mode in ("average", "projection")}
    assert err["projection"] < 0.6 * err["average"], err                          # measured 0.43


def test_mapBackProjectionOnAPatchDropsThinSupport():
    source = planeSurface((0.0, 0.0), (2.0, 2.0), (8, 8), cellType="triangle")
    patch = planeSurface((0.45, 0.55), (1.45, 1.55), (12, 12), cellType="triangle")
    lin = lambda x: 500.0 + 100.0 * x[:, 0]                                        # noqa: E731
    m = SurfaceFluxMapper(source, patch, unmatched="drop")
    res = m.mapBack(lin(patch.points), "point", nodal="projection", minSupport=0.25)
    avg = m.mapBack(lin(patch.points), "point")
    assert res.diagnostics["projectionDroppedNodes"] > 0
    np.testing.assert_array_equal(np.isfinite(res.pointValues), np.isfinite(avg.pointValues))   # same cover
    thin = np.isclose(res.pointValues, avg.pointValues) & ~np.isclose(avg.pointValues, lin(source.points))
    exact = np.isfinite(res.pointValues) & ~thin
    assert thin.sum() > 0 and exact.sum() > 0
    np.testing.assert_allclose(res.pointValues[exact], lin(source.points[exact]), rtol=1e-9)


def test_mapBackReportsHowMuchOfEachPointIsCovered():
    # points on the edge of a patch take the value of the covered part of their support (a nearby
    # location): pointCoverage lets the caller tell them from the fully covered ones
    source = planeSurface((0.0, 0.0), (2.0, 2.0), (8, 8), cellType="triangle")
    patch = planeSurface((0.45, 0.55), (1.45, 1.55), (12, 12), cellType="triangle")
    lin = lambda x: 500.0 + 100.0 * x[:, 0]                                        # noqa: E731
    res = SurfaceFluxMapper(source, patch, unmatched="drop").mapBack(lin(patch.points), "point")
    cov = res.pointCoverage
    assert np.all((cov >= 0.0) & (cov <= 1.0))
    assert np.all(cov[~np.isfinite(res.pointValues)] == 0.0)                       # untouched points
    full, part = cov > 0.999, (cov > 0.0) & (cov < 0.999)
    assert full.any() and part.any()
    x = source.points[:, 0]
    assert np.all((x[full] > 0.45) & (x[full] < 1.45))                             # inside the patch
    err = np.abs(res.pointValues - lin(source.points))
    assert err[part].max() > 10.0 * err[full].max()                                # edge = extrapolated
    np.testing.assert_allclose(SurfaceFluxMapper(source, source).mapBack(lin(source.points), "point").pointCoverage,
                               1.0, rtol=1e-12)
