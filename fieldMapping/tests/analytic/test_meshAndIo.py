"""Finite elements, integration, surfaces, VTU I/O and geometry kernels against closed-form results."""
from math import factorial

import numpy as np
import pytest

from pythonLibs.fieldMapping.Geometry import boxPairs, clipHalfPlane, clipPolygons, closestPointOnTriangles, \
    polygonMoments
from pythonLibs.fieldMapping.io.Vtu import readPvd, readVtu, writePvd, writeVtu
from pythonLibs.fieldMapping.mesh.Elements import _ELEMENTS, element
from pythonLibs.fieldMapping.mesh.Generators import planeSurface, sphereSurface, sphericalShell, structuredBox
from pythonLibs.fieldMapping.mesh.Mesh import UnstructuredMesh

REF_MEASURE = {1: 1, 3: 2, 21: 2, 5: 0.5, 22: 0.5, 9: 4, 23: 4, 28: 4, 10: 1 / 6, 24: 1 / 6, 12: 8, 25: 8, 13: 1.0,
               26: 1.0, 14: 4 / 3}


# ---------------------------------------------------------------- elements
@pytest.mark.parametrize("vtkType", sorted(t for t in _ELEMENTS if t != 1))
def test_elementShapeFunctions(vtkType):
    e = element(vtkType)
    np.testing.assert_allclose(e.shape(e.ref), np.eye(e.nNodes), atol=1e-14)              # Kronecker
    p, w = e.quadrature(6)
    np.testing.assert_allclose(e.shape(p).sum(axis=1), 1.0, atol=1e-14)                    # partition of unity
    np.testing.assert_allclose(e.shape(p) @ e.ref, p, atol=1e-14)                          # linear completeness
    h = 1e-6
    fd = np.stack([(e.shape(p + h * np.eye(e.dim)[k]) - e.shape(p - h * np.eye(e.dim)[k])) / (2 * h)
                   for k in range(e.dim)], axis=2)
    np.testing.assert_allclose(e.dshape(p), fd, atol=1e-8)
    assert w.sum() == pytest.approx(REF_MEASURE[vtkType], rel=1e-13)


def test_simplexQuadratureIsExact():
    p, w = element(5).quadrature(8)
    for a in range(9):
        for b in range(9 - a):
            exact = factorial(a) * factorial(b) / factorial(a + b + 2)
            assert np.sum(w * p[:, 0] ** a * p[:, 1] ** b) == pytest.approx(exact, abs=1e-15)
    p, w = element(10).quadrature(6)
    for a in range(7):
        for b in range(7 - a):
            for c in range(7 - a - b):
                exact = factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)
                assert np.sum(w * p[:, 0] ** a * p[:, 1] ** b * p[:, 2] ** c) == pytest.approx(exact, abs=1e-15)
    p, w = element(14).quadrature(6)                                   # pyramid: int z^2 = 4 / 30
    assert np.sum(w * p[:, 2] ** 2) == pytest.approx(4 / 30, rel=1e-13)


# ---------------------------------------------------------------- integration
@pytest.mark.parametrize("cellType", ["hexahedron", "tetra", "wedge", "quadraticHexahedron", "polyhedron"])
def test_boxIntegrationIsExact(cellType):
    m = structuredBox((0, 0, 0), (2, 1, 3), (3, 2, 2), cellType)
    q = m.quadrature(4)
    f = 1 + m.points[:, 0] - 2 * m.points[:, 1] + 0.5 * m.points[:, 2]
    assert q.weight.sum() == pytest.approx(6.0, rel=1e-13)
    assert q.integrate(f) == pytest.approx(6.0 * (1 + 1 - 1 + 0.75), rel=1e-13)
    surf = m.boundarySurface()
    assert surf.cellMeasures().sum() == pytest.approx(2 * (2 + 6 + 3), rel=1e-13)
    # consistent loads of a unit field are the nodal volumes and sum to the volume
    assert q.loadVector(np.ones(q.weight.size)).sum() == pytest.approx(6.0, rel=1e-13)


def test_quadraticGeometryIsHigherOrder():
    exact = 4 / 3 * np.pi * (1 - 0.8 ** 3)
    err = {}
    for quad in (False, True):
        err[quad] = [abs(sphericalShell((0.8, 1.0), n=n, layers=(1,), quadratic=quad).cellMeasures(order=6).sum()
                         - exact) for n in (4, 8)]
    assert err[True][1] < 0.02 * err[False][1]
    assert np.log2(err[True][0] / err[True][1]) > 3.5                      # O(h^4) volume error


def test_nonConvexPolyhedronAndOrientation():
    # L-shaped prism with inconsistently oriented input faces
    l2 = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    pts = np.vstack([np.column_stack([l2, np.zeros(6)]), np.column_stack([l2, np.ones(6)])])
    faces = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]] + [[i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6] for i in range(6)]
    faces[3] = faces[3][::-1]
    m = UnstructuredMesh.fromBlocks(pts, [], polyhedra=[faces])
    q = m.quadrature(2)
    assert q.weight.sum() == pytest.approx(3.0, rel=1e-13)
    assert q.integrate(pts[:, 0]) == pytest.approx(2.5, rel=1e-13)          # int x: 2 (lower bar) + 0.5 (upper)
    area = m.boundarySurface().cellMeasures().sum()
    assert area == pytest.approx(2 * 3 + 8, rel=1e-13)


def test_surfacesAreOutwardAndInterfacesFound():
    for quad in (False, True):
        m = sphericalShell((0.8, 0.9, 1.0), n=4, layers=(2, 1), quadratic=quad)
        b = m.boundarySurface()
        tri, _ = b.triangulate()
        p = b.points
        nrm = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        c = p[tri].mean(axis=1)
        outer = np.linalg.norm(c, axis=1) > 0.95
        s = np.einsum("ij,ij->i", nrm, c)
        assert np.all(s[outer] > 0) and np.all(s[~outer] < 0)
        itf = m.interfaceSurface(m.cellData["layer"])
        assert set(np.unique(itf.cellData["labelBelow"])) == {0} and set(np.unique(itf.cellData["labelAbove"])) == {1}
    # inverted element: faces still outward
    box = structuredBox(n=(1, 1, 1))
    box.connectivity = box.connectivity[[4, 5, 6, 7, 0, 1, 2, 3]]
    b = box.boundarySurface()
    tri, _ = b.triangulate()
    p = b.points
    nrm = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    assert np.all(np.einsum("ij,ij->i", nrm, p[tri].mean(axis=1) - 0.5) > 0)


# ---------------------------------------------------------------- VTU
def _mixed():
    a = structuredBox((0, 0, 0), (1, 1, 1), (2, 1, 1), "hexahedron")
    b = structuredBox((2, 0, 0), (3, 1, 1), (1, 1, 1), "polyhedron")
    c = planeSurface((4, 0), (5, 1), (2, 2), "polygon")
    d = structuredBox((6, 0, 0), (7, 1, 1), (1, 1, 1), "quadraticHexahedron")
    m = UnstructuredMesh.merge([a, b, c, d])
    rng = np.random.default_rng(0)
    m.pointData["T"] = rng.normal(size=m.nPoints)
    m.pointData["q"] = rng.normal(size=(m.nPoints, 3)).astype(np.float32)
    m.cellData["id"] = np.arange(m.nCells, dtype=np.int32)
    return m


@pytest.mark.parametrize("fmt", ["appended", "binary", "ascii"])
@pytest.mark.parametrize("compress", [True, False])
def test_vtuRoundTrip(tmp_path, fmt, compress):
    m = _mixed()
    path = str(tmp_path / "m.vtu")
    writeVtu(path, m, fmt, compress)
    r = readVtu(path)
    np.testing.assert_array_equal(r.points, m.points)
    np.testing.assert_array_equal(r.connectivity, m.connectivity)
    np.testing.assert_array_equal(r.cellTypes, m.cellTypes)
    for c, faces in m.polyFaces.items():
        assert all(np.array_equal(a, b) for a, b in zip(r.polyFaces[c], faces))
    np.testing.assert_array_equal(r.pointData["q"], m.pointData["q"])
    assert r.pointData["q"].dtype == np.float32 and r.cellData["id"].dtype == np.int32
    np.testing.assert_allclose(r.cellMeasures(), m.cellMeasures(), rtol=1e-14)


def test_newPolyhedronLayoutAndPvd(tmp_path):
    xml = """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="2.2" byte_order="LittleEndian" header_type="UInt64">
<UnstructuredGrid><Piece NumberOfPoints="9" NumberOfCells="2">
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">
0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1 2 0 0</DataArray></Points>
<Cells>
<DataArray type="Int64" Name="connectivity" format="ascii">0 1 2 3 4 5 6 7 1 8 2 5</DataArray>
<DataArray type="Int64" Name="offsets" format="ascii">8 12</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">42 10</DataArray>
<DataArray type="Int64" Name="face_connectivity" format="ascii">0 3 2 1 4 5 6 7 0 1 5 4 1 2 6 5 2 3 7 6 3 0 4 7</DataArray>
<DataArray type="Int64" Name="face_offsets" format="ascii">4 8 12 16 20 24</DataArray>
<DataArray type="Int64" Name="polyhedron_to_faces" format="ascii">0 1 2 3 4 5</DataArray>
<DataArray type="Int64" Name="polyhedron_offsets" format="ascii">6 6</DataArray>
</Cells></Piece></UnstructuredGrid></VTKFile>"""
    path = tmp_path / "new.vtu"
    path.write_text(xml)
    m = readVtu(str(path))
    assert len(m.polyFaces[0]) == 6
    np.testing.assert_allclose(m.cellMeasures(), [1.0, 1 / 6])
    files = []
    for k, t in enumerate((0.0, 0.5, 1.25)):
        f = str(tmp_path / f"s{k}.vtu")
        writeVtu(f, m)
        files.append((t, f))
    writePvd(str(tmp_path / "series.pvd"), files)
    back = readPvd(str(tmp_path / "series.pvd"))
    assert [d["time"] for d in back] == [0.0, 0.5, 1.25]
    assert readVtu(back[2]["file"]).nCells == 2


# ---------------------------------------------------------------- geometry kernels
def test_clippingAndMoments():
    # unit square clipped by the triangle (0,0) (2,0) (0,2): area 1, centroid (1/2, 1/2)
    sq = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], float)
    poly, n, att = clipPolygons(sq, np.array([4]), np.zeros((1, 4, 0)), np.array([[[0, 0], [2, 0], [0, 2]]], float))
    area, cen = polygonMoments(poly, n)
    assert area[0] == pytest.approx(1.0) and np.allclose(cen[0], 0.5)
    # triangle (0,0) (1,0) (0,1) cut by x <= 0.5: area 3/8
    tri = np.array([[[0, 0], [1, 0], [0, 1]]], float)
    p2, n2, _, _ = clipHalfPlane(tri, np.array([3]), np.zeros((1, 3, 0)), 0.5 - tri[..., 0])
    assert polygonMoments(p2, n2)[0][0] == pytest.approx(3 / 8)
    # vertex exactly on the line produces no duplicate vertex
    p3, n3, _, _ = clipHalfPlane(tri, np.array([3]), np.zeros((1, 3, 0)), -tri[..., 1])
    assert n3[0] == 2                                          # degenerate (zero area) edge on the line


def test_boxPairsAndClosestPoints():
    rng = np.random.default_rng(1)
    loA = rng.uniform(0, 10, (800, 3))
    hiA = loA + rng.uniform(0, 0.6, (800, 3))
    loB = rng.uniform(0, 10, (700, 3))
    hiB = loB + rng.uniform(0, 0.9, (700, 3))
    hiB[0] = loB[0] + 15                                      # one oversized box
    i, j = boxPairs(loA, hiA, loB, hiB)
    brute = np.argwhere(np.all((loA[:, None] <= hiB[None]) & (hiA[:, None] >= loB[None]), axis=2))
    assert set(zip(i.tolist(), j.tolist())) == set(map(tuple, brute.tolist()))
    tri = rng.normal(size=(300, 3, 3))
    p = 2 * rng.normal(size=(300, 3))
    q, d, bary = closestPointOnTriangles(p, tri)
    u = rng.uniform(size=(20000, 2))
    u = np.where(u.sum(axis=1, keepdims=True) > 1, 1 - u, u)
    samples = np.einsum("sk,mkd->msd", np.column_stack([1 - u.sum(axis=1), u]), tri)
    assert np.all(d <= np.linalg.norm(samples - p[:, None], axis=2).min(axis=1) + 1e-12)
    assert bary.min() >= -1e-12 and np.allclose(np.einsum("mk,mkd->md", bary, tri), q)


def test_sphereSurfaceArea():
    for kind in ("quad", "triangle"):
        errs = [abs(sphereSurface(1.0, n, kind).cellMeasures().sum() - 4 * np.pi) for n in (8, 16)]
        assert np.log2(errs[0] / errs[1]) > 1.8                  # flat facets: O(h^2)
