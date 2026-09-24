# fieldMapping

Conservative transfer of field data between non-matching meshes, built on numpy and
the Python standard library only.

- **VTU / PVD** reading and writing (`io`): every encoding VTK writes (ascii, inline
  binary, appended raw / base64, zlib or lzma compression, 32 / 64-bit headers,
  several pieces) and arbitrary polyhedra in both the classic `faces` layout and the
  newer face-connectivity layout.
- **Unstructured meshes** (`mesh`): linear and quadratic VTK cells (line, triangle,
  quad, tetra, hexahedron, wedge, pyramid; quadratic edge, triangle, quad, tetra,
  hexahedron, wedge; biquadratic quad), polygons and polyhedra (also non-convex; face
  orientation is repaired). Finite-element integration with exact quadrature on the
  curved geometry, consistent load vectors, boundary and material-interface surfaces
  with outward orientation (inverted cells handled).
- **Conservative surface flux transfer** (`SurfaceFluxMapper`): common refinement of
  the two surfaces, incoming and outgoing heat conserved separately to round-off,
  no heat of the wrong sign, automatic normal orientation, cell / nodal-load / nodal
  flux outputs.

## Layout

```text
fieldMapping/
  io/Vtu.py               readVtu, writeVtu, readPvd, writePvd
  mesh/Elements.py        VTK cell library: shape functions, faces, quadrature (collapsed Gauss-Jacobi on simplices)
  mesh/Mesh.py            UnstructuredMesh, Quadrature, boundarySurface, interfaceSurface, triangulate
  mesh/Generators.py      structuredBox, planeSurface, sphereSurface, sphericalShell
  Geometry.py             box pair search, polygon clipping, closest points
  SurfaceFluxMapper.py    conservative, sign-preserving surface flux transfer
  tests/analytic/         closed-form verification
```

## Meshes and integration

```python
from pythonLibs.fieldMapping import readVtu, writeVtu

mesh = readVtu("structure.vtu")             # any cell types, polyhedra included
q = mesh.quadrature(order=4)                # points, physical weights, interpolation operator
volume = q.weight.sum()
energy = q.integrate(nodalField)            # = weights @ (interp @ nodalField)
loads = q.loadVector(valuesAtPoints)        # consistent nodal loads, int N_i f
outer = mesh.boundarySurface()              # outward faces, indexed by the volume points
layers = mesh.interfaceSurface(mesh.cellData["material"])
writeVtu("out.vtu", mesh)                   # appended raw + zlib by default
```

Polygons are integrated as triangle fans around their centroid and polyhedra as
tetrahedra (cell centroid, face centroid, edge): any linear field is reproduced
exactly and every closed polyhedron gets its exact volume (signed decomposition,
also for non-convex cells).

## Conservative surface flux transfer

```python
from pythonLibs.fieldMapping import SurfaceFluxMapper, readVtu

flow = readVtu("surfaceFlux.vtu")           # flux on facets (cell data) or nodes (point data)
structure = readVtu("structure.vtu")        # volume mesh: its boundary surface is the target
mapper = SurfaceFluxMapper(flow, structure) # geometry built once
res = mapper.map(flow.cellData["q"], "cell", reconstruction="linear", pointFlux=True)
res.diagnostics                             # heat in / out per side, dropped heat, peaks, coverage
res.nodalLoads                              # consistent nodal loads on the structure's nodes
res.attach(mapper.target)                   # cell flux, heat in / out, nodal loads, nodal flux
```

- the source heat of every source triangle, split into its incoming (q > 0) and
  outgoing (q < 0) parts along the zero line of q, is distributed to the target
  facets in proportion to the integrals of q over their overlaps: incoming and
  outgoing heat are conserved separately and no facet gets heat of the wrong sign
- gaps (no overlap) go to the nearest facet within `maxDistance`; heat farther away
  (a target covering only part of the source) is reported as dropped
- `reconstruction="linear"` (cell data) uses a limited least-squares gradient that
  keeps the facet mean, the neighbour bounds and the sign: sharper coarse-to-fine
  transfer with the same conservation
- `pointFlux=True` adds nodal flux values: lumped L2 projection corrected by the
  conservative projection of `regressionHandler.constraints` (incoming and outgoing
  parts separately with sign bounds, optional per-group heat constraints)
- the overlap geometry is computed once; `map` can be called for every time step

## Tests

```bash
python -m pytest fieldMapping/tests -q
```

`tests/analytic/` checks shape functions and quadrature exactness, exact volumes and
areas (also non-convex polyhedra), fourth-order geometry of quadratic cells, VTU round
trips, clipping and search kernels, and for the flux transfer: exact reproduction on
identical meshes, conservation per sign to round-off, second-order convergence under
refinement, partial targets and linear reconstruction.
