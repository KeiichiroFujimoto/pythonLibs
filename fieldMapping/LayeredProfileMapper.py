"""Energy-conserving mapping of layered 1-D through-thickness profiles onto a 3-D mesh.

Input: a set of *stations* (points on the outer surface where a 1-D layered
model was solved), each with a depth grid, the depths of its layer
interfaces and temperature histories; and a 3-D mesh whose cells carry a
layer label (outermost layer first).

1. Depth coordinate. For a node in layer l, xi = a / (a + b) with a, b its
   distances to the upper and lower surfaces of that layer in the 3-D mesh
   (outer surface, material interfaces, inner surface). Each station maps the
   node to its own depth in the same layer so that layer boundaries always
   match, even when the 3-D and 1-D thicknesses differ (surface recession):
   - "surfaceAnchored" (default): in the outer layer
     z_k = a (1 - xi) + xi^2 d_k (d_k the station's outer-layer thickness):
     the absolute depth below the current surface near the surface (where the
     heating sets the profile), the station's interface at xi = 1; inner
     layers use the layer-normalized depth. Falls back to the normalized depth
     where the station layer is less than half as thick as the local one.
   - "layerNormalized": z_k = b_k,l + xi (b_k,l+1 - b_k,l) in every layer.
   - "depth": the distance to the outer surface.
2. Surface interpolation between stations with weights w_k(x) of the node's
   foot point on the outer surface: ordinary Kriging (default; Matern 5/2 with
   the correlation length fitted to the stations' peak surface temperatures),
   inverse distance or nearest station. The same weights are used at every
   depth of a column, so the through-thickness profile stays smooth.
3. Reference energy. The interpolated continuous field
   T*(x) = sum_k w_k(x) T_k(z_k(xi(x))) is integrated with a high-order rule,
   E_ref,g = int_g rho e(T*) dV, per conservation group g (station tributary
   x layer by default). This resolves steep 1-D gradients that a coarse 3-D
   mesh cannot represent nodally.
4. Conservation. Nodal temperatures start from T* at the nodes and are
   corrected by the smallest change in the heat-capacity metric that makes
   the finite-element energy E_h,g(T) = int_g rho e(N T) dV equal E_ref,g
   (nonlinear through cp(T); sequential linearization), with every node kept
   inside the range of the station values around its depth (no new extrema).

The result per time step: nodal temperatures, cell temperatures (the value
whose energy equals the cell's finite-element energy), and per group the
reference, nodal-interpolation and final energies (relative errors of the final energy
are at round-off level).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from pythonLibs.fieldMapping.Geometry import SurfaceLocator
from pythonLibs.fieldMapping.Material import Material
from pythonLibs.fieldMapping.mesh.Mesh import UnstructuredMesh
from pythonLibs.regressionHandler.constraints.Projection import projectNonlinear
from pythonLibs.regressionHandler.numerics.NeighborSearch import NeighborSearch
from pythonLibs.regressionHandler.numerics.Sparse import SparseMatrix


# ---------------------------------------------------------------- stations
class Station:
    """One 1-D layered profile history.

    Args:
        position:    (3,) point on (or near) the outer surface
        time:        (nt,) increasing times
        depth:       (nz,) depths below the current surface, or (nt, nz)
        temperature: (nt, nz)
        interfaces:  (L + 1,) layer boundary depths [0, b1, ..., total], or (nt, L + 1)
    """

    def __init__(self, position, time, depth, temperature, interfaces, name: str = "") -> None:
        self.position = np.asarray(position, dtype=float).ravel()
        self.time = np.atleast_1d(np.asarray(time, dtype=float))
        nt = self.time.size
        self.temperature = np.atleast_2d(np.asarray(temperature, dtype=float))
        if self.temperature.shape[0] != nt:
            raise ValueError("temperature needs one row per time")
        d = np.asarray(depth, dtype=float)
        self.depth = np.broadcast_to(d, self.temperature.shape).copy() if d.ndim == 1 else d
        b = np.asarray(interfaces, dtype=float)
        self.interfaces = np.broadcast_to(b, (nt, b.shape[-1])).copy() if b.ndim == 1 else b
        if self.depth.shape != self.temperature.shape or self.interfaces.shape[0] != nt:
            raise ValueError("depth / interfaces do not match the time and depth grids")
        if nt > 1 and np.any(np.diff(self.time) <= 0):
            raise ValueError("station times must increase")
        if np.any(np.diff(self.depth, axis=1) <= 0) or np.any(np.diff(self.interfaces, axis=1) <= 0):
            raise ValueError("depths and interface depths must increase")
        self.name = name
        self.nLayers = self.interfaces.shape[1] - 1

    @classmethod
    def fromSpec(cls, spec) -> "Station":
        if isinstance(spec, Station):
            return spec
        return cls(spec["position"], spec["time"], spec["depth"], spec["temperature"], spec["interfaces"],
                   spec.get("name", ""))

    def at(self, t: float):
        """(depth, temperature, interfaces) at time t (linear interpolation, clamped)."""
        tt = self.time
        if tt.size == 1 or t <= tt[0]:
            return self.depth[0], self.temperature[0], self.interfaces[0]
        if t >= tt[-1]:
            return self.depth[-1], self.temperature[-1], self.interfaces[-1]
        k = int(np.searchsorted(tt, t) - 1)
        a = (t - tt[k]) / (tt[k + 1] - tt[k])
        mix = lambda v: (1 - a) * v[k] + a * v[k + 1]            # noqa: E731
        return mix(self.depth), mix(self.temperature), mix(self.interfaces)


@dataclass
class LayeredMapResult:
    """Mapped nodal temperatures per time and conservation diagnostics."""
    times: np.ndarray
    temperature: np.ndarray               # (nt, nPoints), NaN outside the layered cells
    diagnostics: list = field(default_factory=list)
    cellTemperature: Optional[np.ndarray] = None     # (nt, nCells): energy-equivalent cell values

    def summary(self) -> dict:
        errs = [d["maxRelativeEnergyError"] for d in self.diagnostics]
        return {"nTimes": int(self.times.size), "maxRelativeEnergyError": float(max(errs)) if errs else 0.0,
                "maxRelativeNodalEnergyError": float(max(d["maxRelativeNodalEnergyError"] for d in self.diagnostics))
                if self.diagnostics else 0.0,
                "converged": all(d["converged"] for d in self.diagnostics)}


# ---------------------------------------------------------------- mapper
class LayeredProfileMapper:
    """Map layered 1-D profiles onto a 3-D mesh with exact energy conservation (see the module docstring).

    Args:
        mesh:          3-D UnstructuredMesh with a layer label per cell
        stations:      list of Station objects or specs
        materials:     {layer label: Material or {"density": ..., "cp": ... }} (3-D side properties)
        layerArray:    name of the cell array with the layer labels
        layers:        labels ordered from the outer surface inward (default: sorted labels present in materials)
        outerSurface:  optional surface mesh of the heated surface (default: detected, see ``_outerFaces``)
        weights:       "kriging" (default), "idw" or "nearest"
        nearest:       number of stations used by "idw"
        power:         inverse-distance power
        lengthscale:   Kriging correlation length (default "auto": maximum likelihood on the stations'
                       peak surface temperatures; a number fixes it)
        depthMode:     "surfaceAnchored" (default), "layerNormalized" or "depth"
        conservation:  "stationLayer" (default), "layer", "global" or "cell"
        order:         quadrature order of the finite-element energy
        referenceOrder: quadrature order of the reference energy of the continuous field
        bounds:        keep nodal values within the local range of the station values
        maxAngle:      angle (degrees) separating outer / inner faces from lateral faces
    """

    def __init__(self, mesh: UnstructuredMesh, stations: Sequence, materials: dict, layerArray: str = "layer",
                 layers: Optional[Sequence] = None, outerSurface: Optional[UnstructuredMesh] = None,
                 weights: str = "kriging", nearest: int = 4, power: float = 2.0, lengthscale="auto",
                 depthMode: str = "surfaceAnchored", conservation: str = "stationLayer", order: int = 4,
                 referenceOrder: int = 10, bounds: bool = True, maxAngle: float = 60.0) -> None:
        if weights not in ("idw", "kriging", "nearest"):
            raise ValueError("weights must be idw, kriging or nearest")
        if depthMode not in ("surfaceAnchored", "layerNormalized", "depth"):
            raise ValueError("depthMode must be surfaceAnchored, layerNormalized or depth")
        if conservation not in ("stationLayer", "layer", "global", "cell"):
            raise ValueError("conservation must be stationLayer, layer, global or cell")
        self.mesh = mesh
        self.stations = [Station.fromSpec(s) for s in stations]
        if not self.stations:
            raise ValueError("at least one station is required")
        mats = {k: Material.fromSpec(v) for k, v in materials.items()}
        labels = np.asarray(mesh.cellData[layerArray])
        self.layers = list(layers) if layers is not None else sorted(mats, key=lambda v: v)
        missing = [l for l in self.layers if l not in mats]
        if missing:
            raise ValueError(f"no material for layers {missing}")
        L = len(self.layers)
        if any(s.nLayers != L for s in self.stations):
            raise ValueError(f"every station needs {L} layers (interfaces of length {L + 1})")
        self.materials = [mats[l] for l in self.layers]
        pos = {l: i for i, l in enumerate(self.layers)}
        self.cellLayer = np.array([pos.get(v, -1) for v in labels.tolist()], dtype=np.int64)
        self.cells = np.flatnonzero(self.cellLayer >= 0)
        if self.cells.size == 0:
            raise ValueError("no cells carry the given layer labels")
        self.options = {"weights": weights, "nearest": nearest, "power": power, "lengthscale": lengthscale,
                        "depthMode": depthMode, "conservation": conservation, "order": order,
                        "referenceOrder": referenceOrder, "bounds": bounds, "maxAngle": maxAngle}
        self._cosAngle = np.cos(np.radians(maxAngle))
        self._buildSurfaces(outerSurface)
        self._buildNodes()
        self._buildQuadrature()

    # ------------------------------------------------------------------ geometry
    def _layerNodes(self, l: int) -> np.ndarray:
        cells = self.cells[self.cellLayer[self.cells] == l]
        return np.unique(np.concatenate([self.mesh.cell(c) for c in cells])) if cells.size else np.zeros(0, np.int64)

    def _buildSurfaces(self, outerSurface) -> None:
        mesh, L = self.mesh, len(self.layers)
        bnd = mesh.boundarySurface(self.cells)
        ownerLayer = self.cellLayer[bnd.cellData["ownerCell"]]
        itf = mesh.interfaceSurface(np.where(self.cellLayer >= 0, self.cellLayer, -1), self.cells) if L > 1 else None
        self._interfaces = []
        for l in range(L - 1):
            sel = np.flatnonzero((itf.cellData["labelBelow"] == l) & (itf.cellData["labelAbove"] == l + 1))
            if sel.size == 0:
                raise ValueError(f"layers {self.layers[l]} and {self.layers[l + 1]} share no faces")
            self._interfaces.append(SurfaceLocator.fromMesh(itf, sel))
        # outer surface
        if outerSurface is not None:
            self._outer = SurfaceLocator.fromMesh(outerSurface)
        else:
            cand = np.flatnonzero(ownerLayer == 0)
            self._outer = SurfaceLocator.fromMesh(bnd, self._alignedFaces(bnd, cand, upper=True))
        # inner surface of the last layer: boundary faces pointing away from the surface above
        cand = np.flatnonzero(ownerLayer == L - 1)
        inner = self._alignedFaces(bnd, cand, upper=False)
        self._inner = SurfaceLocator.fromMesh(bnd, inner) if inner.size else None

    def _alignedFaces(self, bnd: UnstructuredMesh, cand: np.ndarray, upper: bool) -> np.ndarray:
        """Faces among ``cand`` whose outward normal points away from the neighbouring layer surface.

        Outer faces of layer 0 face away from the first interface (or, for a
        single layer, along the normals of the stations); inner faces of the
        last layer face away from the surface above it. Lateral faces are
        excluded by the angle test.
        """
        if cand.size == 0:
            return cand
        tri, par = bnd.triangulate()
        keep = np.isin(par, cand)
        tri, par = tri[keep], par[keep]
        p = bnd.points[tri]
        cen = p.mean(axis=1)
        nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-300)[:, None]
        if upper:
            if self._interfaces:
                _, cp, _, _ = self._interfaces[0].closest(cen)
                ref = cen - cp
            else:
                # single layer: the outer faces are those closest to the stations (their normals define "out")
                loc = SurfaceLocator(p, par)
                _, _, tid, _ = loc.closest(np.array([s.position for s in self.stations]))
                sn = loc.normals(tid)
                spos = np.array([s.position for s in self.stations])
                near = NeighborSearch(spos).query(cen, 1)[1][:, 0]
                ref = sn[near]
        else:
            above = self._interfaces[-1] if self._interfaces else None
            if above is None:
                return self._alignedFacesSingleInner(bnd, cand, tri, par, cen, nrm)
            _, cp, _, _ = above.closest(cen)
            ref = cen - cp
        ref /= np.maximum(np.linalg.norm(ref, axis=1), 1e-300)[:, None]
        ok = np.einsum("ij,ij->i", nrm, ref) > self._cosAngle
        good = np.zeros(bnd.nCells, dtype=bool)
        np.logical_or.at(good, par, ok)
        return np.intersect1d(cand, np.flatnonzero(good))

    def _alignedFacesSingleInner(self, bnd, cand, tri, par, cen, nrm):
        _, cp, _, _ = self._outer.closest(cen)
        ref = cen - cp
        dist = np.linalg.norm(ref, axis=1)
        ref = ref / np.maximum(dist, 1e-300)[:, None]
        ok = (np.einsum("ij,ij->i", nrm, ref) > self._cosAngle) & (dist > 1e-9 * max(1.0, float(dist.max())))
        good = np.zeros(bnd.nCells, dtype=bool)
        np.logical_or.at(good, par, ok)
        return np.intersect1d(cand, np.flatnonzero(good))

    def _buildNodes(self) -> None:
        mesh, L = self.mesh, len(self.layers)
        nodes = np.unique(np.concatenate([mesh.cell(c) for c in self.cells]))
        self.nodes = nodes
        pts = mesh.points[nodes]
        # foot points on the outer surface
        dOut, foot, _, _ = self._outer.closest(pts)
        self._footDepth = dOut
        self._foot = foot
        # station positions snapped to the outer surface
        spos = np.array([s.position for s in self.stations])
        _, self.stationFoot, _, _ = self._outer.closest(spos)
        self.lengthscale = self._fitLengthscale() if self.options["weights"] == "kriging" else None
        self._nodeWeights = self._weights(foot)                       # SparseMatrix (nNodes, nStations)
        # layer-normalized depth per layer (NaN where the node is not in the layer)
        local = {int(n): i for i, n in enumerate(nodes)}
        self._xi = np.full((L, nodes.size), np.nan)
        for l in range(L):
            ln = self._layerNodes(l)
            idx = np.array([local[int(n)] for n in ln], dtype=np.int64)
            p = mesh.points[ln]
            a = dOut[idx] if l == 0 else self._interfaces[l - 1].closest(p)[0]
            if l < L - 1:
                b = self._interfaces[l].closest(p)[0]
            elif self._inner is not None:
                b = self._inner.closest(p)[0]
            else:
                b = None
            if self.options["depthMode"] == "depth" or b is None:
                self._xi[l, idx] = np.nan                            # depth mode for this layer
            else:
                self._xi[l, idx] = a / np.maximum(a + b, 1e-300)
        self._primaryLayer = np.argmax(~np.isnan(self._xi) | self._inLayer(), axis=0)

    def _inLayer(self) -> np.ndarray:
        L = len(self.layers)
        out = np.zeros((L, self.nodes.size), dtype=bool)
        local = np.full(self.mesh.nPoints, -1, dtype=np.int64)
        local[self.nodes] = np.arange(self.nodes.size)
        for l in range(L):
            out[l, local[self._layerNodes(l)]] = True
        return out

    def _weights(self, foot: np.ndarray) -> SparseMatrix:
        """Station weights of foot points (m, 3) -> SparseMatrix (m, nStations), rows summing to 1."""
        spos = self.stationFoot
        ns, m = spos.shape[0], foot.shape[0]
        mode = self.options["weights"]
        if mode == "nearest" or ns == 1:
            idx = NeighborSearch(spos).query(foot, 1)[1][:, 0]
            return SparseMatrix(np.arange(m), idx, np.ones(m), (m, ns))
        if mode == "idw":
            k = min(int(self.options["nearest"]), ns)
            d, idx = NeighborSearch(spos).query(foot, k)
            scale = max(float(np.median(d[:, -1])), 1e-300)
            exact = d[:, 0] <= 1e-12 * scale
            w = 1.0 / np.maximum(d / scale, 1e-12) ** float(self.options["power"])     # exact hits handled below
            w = np.where(exact[:, None], (np.arange(k) == 0)[None, :].astype(float), w)
            w /= w.sum(axis=1, keepdims=True)
            return SparseMatrix(np.repeat(np.arange(m), k), idx.ravel(), w.ravel(), (m, ns))
        # ordinary Kriging with a Matern 5/2 correlation
        ell = self.lengthscale
        corr = lambda r: (1 + np.sqrt(5) * r + 5 * r * r / 3) * np.exp(-np.sqrt(5) * r)    # noqa: E731
        kss = corr(np.linalg.norm(spos[:, None] - spos[None], axis=2) / ell) + 1e-10 * np.eye(ns)
        big = np.block([[kss, np.ones((ns, 1))], [np.ones((1, ns)), np.zeros((1, 1))]])
        rows = []
        for s in range(0, m, 20000):
            f = foot[s:s + 20000]
            kxs = corr(np.linalg.norm(f[:, None] - spos[None], axis=2) / ell)
            rhs = np.hstack([kxs, np.ones((f.shape[0], 1))])
            rows.append(np.linalg.solve(big, rhs.T).T[:, :ns])
        w = np.vstack(rows)
        r, c = np.nonzero(np.abs(w) > 1e-14)
        return SparseMatrix(r, c, w[r, c], (m, ns))

    def _fitLengthscale(self) -> float:
        """Correlation length for the Kriging weights (fixed, or maximum likelihood on the stations)."""
        spos = self.stationFoot
        opt = self.options["lengthscale"]
        dd = NeighborSearch(spos).query(spos, 2)[0][:, 1] if spos.shape[0] > 1 else np.ones(1)
        fallback = 2.0 * float(np.median(dd))
        if opt not in (None, "auto"):
            return float(opt)
        if spos.shape[0] < 4:
            return fallback
        # peak surface temperature of every station: the quantity whose spatial pattern the weights carry
        y = np.array([float(np.max(s.temperature[:, 0])) for s in self.stations])
        if np.ptp(y) <= 1e-9 * max(1.0, float(np.abs(y).max())):
            return fallback
        try:
            from pythonLibs.regressionHandler import KrigingModel
            m = KrigingModel(corr={"type": "matern52", "ard": False}, poly="constant", normalize=False).fit(spos, y)
            ell = float(np.exp(m._kp[0]))
        except Exception:
            return fallback
        extent = float(np.max(np.linalg.norm(spos - spos.mean(axis=0), axis=1))) * 2 + 1e-300
        return float(np.clip(ell, 0.5 * float(np.min(dd)), 10.0 * extent)) if np.isfinite(ell) else fallback

    # ------------------------------------------------------------------ quadrature and groups
    def _buildQuadrature(self) -> None:
        mesh = self.mesh
        self._qFE = mesh.quadrature(self.options["order"], self.cells)
        self._qRef = mesh.quadrature(self.options["referenceOrder"], self.cells)
        # restrict the interpolation operators to the layered nodes
        local = np.full(mesh.nPoints, -1, dtype=np.int64)
        local[self.nodes] = np.arange(self.nodes.size)
        self._local = local
        for q in (self._qFE, self._qRef):
            q.interpLocal = SparseMatrix.raw(q.interp.row, local[q.interp.col], q.interp.data,
                                             (q.interp.shape[0], self.nodes.size))
        # groups
        mode = self.options["conservation"]
        cl = self.cellLayer
        if mode == "global":
            cellGroup = np.zeros(mesh.nCells, dtype=np.int64)
        elif mode == "layer":
            cellGroup = np.maximum(cl, 0)
        elif mode == "cell":
            cellGroup = np.arange(mesh.nCells)
        else:
            # station with the largest weight at the cell (average of its nodal weights), per layer
            W = self._nodeWeights
            cells = self.cells
            sizes = mesh.offsets[cells + 1] - mesh.offsets[cells]
            owner = np.repeat(np.arange(cells.size), sizes)
            nodeIdx = local[np.concatenate([mesh.cell(c) for c in cells])]
            dense = np.zeros((cells.size, len(self.stations)))
            per = np.zeros((self.nodes.size, len(self.stations)))
            per[W.row, W.col] = W.data
            np.add.at(dense, owner, per[nodeIdx])
            best = np.argmax(dense, axis=1)
            cellGroup = np.zeros(mesh.nCells, dtype=np.int64)
            cellGroup[cells] = best * len(self.layers) + cl[cells]
        used = np.unique(cellGroup[self.cells])
        remap = {int(g): i for i, g in enumerate(used)}
        self.cellGroup = np.full(mesh.nCells, -1, dtype=np.int64)
        self.cellGroup[self.cells] = [remap[int(g)] for g in cellGroup[self.cells]]
        self.nGroups = used.size
        for q in (self._qFE, self._qRef):
            q.group = self.cellGroup[q.cell]
            q.layer = cl[q.cell]
        # xi and weights at the reference points (interpolated from the nodes of each cell's own layer)
        q = self._qRef
        self._refXi = np.full(q.cell.size, np.nan)
        for l in range(len(self.layers)):
            sel = q.layer == l
            if not sel.any():
                continue
            xiL = self._xi[l]
            val = q.interpLocal @ np.nan_to_num(xiL, nan=0.0)
            # depth mode (NaN at the nodes) stays depth mode at the points
            nanPart = SparseMatrix.raw(q.interpLocal.row, q.interpLocal.col, np.abs(q.interpLocal.data),
                                 q.interpLocal.shape) @ np.isnan(xiL).astype(float)
            self._refXi[sel] = np.where(nanPart > 0, np.nan, val)[sel]
        self._refDepth = q.interpLocal @ self._footDepth
        # station weights at the reference points from their (interpolated) foot points
        self._refWeights = self._weights(q.interpLocal @ self._foot)

    # ------------------------------------------------------------------ evaluation
    def _stationDepth(self, k: int, xi: np.ndarray, depth: np.ndarray, layer: np.ndarray, interfaces: np.ndarray):
        """Station-k depth of points with layer-normalized xi (NaN -> depth mode) in ``layer``.

        ``depth`` is the distance of the point to the outer surface.
        """
        lo, hi = interfaces[layer], interfaces[layer + 1]
        x = np.nan_to_num(xi)
        z = lo + x * (hi - lo)
        if self.options["depthMode"] == "surfaceAnchored":
            # outer layer: absolute depth near the surface, station interface at xi = 1
            dk = hi - lo
            local = np.where(x > 1e-12, depth / np.maximum(x, 1e-12), dk)          # local outer-layer thickness
            anchored = depth * (1.0 - x) + x * x * dk
            use = (layer == 0) & (dk >= 0.5 * local)                                 # monotone in xi
            z = np.where(use, anchored, z)
        z = np.where(np.isnan(xi), depth, z)
        return np.clip(z, interfaces[0], interfaces[-1])

    def _field(self, W: SparseMatrix, xi, depth, layer, profiles) -> np.ndarray:
        """sum_k w_k T_k(z_k) for rows of W."""
        out = np.zeros(W.shape[0])
        for k in np.unique(W.col):
            sel = W.col == k
            rows, w = W.row[sel], W.data[sel]
            d, T, b = profiles[k]
            z = self._stationDepth(k, xi[rows], depth[rows], layer[rows], b)
            out[rows] += w * np.interp(z, d, T)
        return out

    def _nodalField(self, profiles) -> np.ndarray:
        W = self._nodeWeights
        L = len(self.layers)
        layer = np.minimum(self._primaryLayer, L - 1)
        xi = self._xi[layer, np.arange(self.nodes.size)]
        return self._field(W, xi, self._footDepth, layer, profiles)

    def _bounds(self, profiles):
        """Per node: min / max of the contributing station profiles over the depths of its neighbourhood."""
        mesh = self.mesh
        W = self._nodeWeights
        cells = self.cells
        sizes = mesh.offsets[cells + 1] - mesh.offsets[cells]
        owner = np.repeat(np.arange(cells.size), sizes)
        nodeIdx = self._local[np.concatenate([mesh.cell(c) for c in cells])]
        lay = self.cellLayer[cells][owner]
        # use a global depth coordinate per station: depth in station k of the entry (layer, xi)
        lower = np.full(self.nodes.size, np.inf)
        upper = np.full(self.nodes.size, -np.inf)
        for k in np.unique(W.col):
            d, T, b = profiles[k]
            xiE = self._xi[lay, nodeIdx]
            zE = self._stationDepth(k, xiE, self._footDepth[nodeIdx], lay, b)
            cellMin = np.full(cells.size, np.inf)
            cellMax = np.full(cells.size, -np.inf)
            np.minimum.at(cellMin, owner, zE)
            np.maximum.at(cellMax, owner, zE)
            zl = np.full(self.nodes.size, np.inf)
            zh = np.full(self.nodes.size, -np.inf)
            np.minimum.at(zl, nodeIdx, cellMin[owner])
            np.maximum.at(zh, nodeIdx, cellMax[owner])
            rows = W.row[W.col == k]
            zz = zl[rows][:, None] + (zh[rows] - zl[rows])[:, None] * np.linspace(0, 1, 17)[None, :]
            inside = (d[None, :] >= zl[rows][:, None]) & (d[None, :] <= zh[rows][:, None])
            tv = np.interp(zz, d, T)
            tmin = np.minimum(tv.min(axis=1), np.where(inside, T[None, :], np.inf).min(axis=1))
            tmax = np.maximum(tv.max(axis=1), np.where(inside, T[None, :], -np.inf).max(axis=1))
            np.minimum.at(lower, rows, tmin)
            np.maximum.at(upper, rows, tmax)
        span = np.maximum(np.abs(upper - lower), 1.0) * 1e-12
        return lower - span, upper + span

    def _energy(self, q, temperatureAtPoints) -> tuple[np.ndarray, np.ndarray]:
        """Volumetric energy and capacity at quadrature points (per layer material)."""
        e = np.zeros(q.cell.size)
        c = np.zeros(q.cell.size)
        for l, mat in enumerate(self.materials):
            sel = q.layer == l
            e[sel] = mat.volumetricEnergy(temperatureAtPoints[sel])
            c[sel] = mat.volumetricCapacity(temperatureAtPoints[sel])
        return e, c

    # ------------------------------------------------------------------ mapping
    def map(self, times=None, tol: float = 1e-12) -> LayeredMapResult:
        """Nodal temperatures at ``times`` (default: the union of the station times)."""
        if times is None:
            times = np.unique(np.concatenate([s.time for s in self.stations]))
        times = np.atleast_1d(np.asarray(times, dtype=float))
        qF, qR = self._qFE, self._qRef
        G = self.nGroups
        out = np.full((times.size, self.mesh.nPoints), np.nan)
        cellOut = np.full((times.size, self.mesh.nCells), np.nan)
        diags = []
        prev = None
        for it, t in enumerate(times):
            profiles = [s.at(t) for s in self.stations]
            x0 = self._nodalField(profiles)
            tRef = self._field(self._refWeights, self._refXi, self._refDepth, qR.layer, profiles)
            eRef, _ = self._energy(qR, tRef)
            target = np.bincount(qR.group, weights=qR.weight * eRef, minlength=G)

            def constraint(T):
                tq = qF.interpLocal @ T
                e, c = self._energy(qF, tq)
                val = np.bincount(qF.group, weights=qF.weight * e, minlength=G)
                ent = qF.interpLocal
                jac = SparseMatrix(qF.group[ent.row], ent.col, ent.data * (qF.weight * c)[ent.row], (G, T.size))
                return val, jac
            # heat-capacity metric (HRZ-lumped)
            tq0 = qF.interpLocal @ x0
            _, cap = self._energy(qF, tq0)
            ent = qF.interpLocal
            diagE = ent.data ** 2 * (qF.weight * cap)[ent.row]
            cellOf = qF.cell[ent.row]
            cellCap = np.bincount(qF.cell, weights=qF.weight * cap, minlength=self.mesh.nCells)
            sumDiag = np.bincount(cellOf, weights=diagE, minlength=self.mesh.nCells)
            lumped = np.bincount(ent.col, weights=diagE * (cellCap / np.where(sumDiag > 0, sumDiag, 1.0))[cellOf],
                                 minlength=self.nodes.size)
            lumped = np.maximum(lumped, 1e-300)
            lb, ub = self._bounds(profiles) if self.options["bounds"] else (None, None)
            nodalEnergy = constraint(x0)[0]
            # energy scale per group: |E_ref| or the energy of a 1 K change, whichever is larger
            groupCap = np.bincount(qF.group, weights=qF.weight * cap, minlength=G)
            scale = np.maximum(np.abs(target), groupCap)
            res = projectNonlinear(x0, lumped, constraint, target, lb, ub, tol=tol, start=prev, scale=scale)
            prev = res.x
            out[it, self.nodes] = res.x
            cellOut[it] = self._cellTemperature(res.x)
            final = constraint(res.x)[0]
            diags.append({"time": float(t), "converged": bool(res.converged), "iterations": res.iterations,
                          "referenceEnergy": target.tolist(), "nodalInterpolationEnergy": nodalEnergy.tolist(),
                          "finalEnergy": final.tolist(),
                          "maxRelativeEnergyError": float(np.max(np.abs(final - target) / scale)),
                          "maxRelativeNodalEnergyError": float(np.max(np.abs(nodalEnergy - target) / scale)),
                          "maxCorrection": float(np.max(np.abs(res.x - x0))),
                          "nodesAtBounds": int(res.atBounds.sum()) if lb is not None else 0})
        return LayeredMapResult(times, out, diags, cellOut)

    def _cellTemperature(self, T: np.ndarray) -> np.ndarray:
        """Per cell the temperature whose energy rho e(T_c) V_c equals the cell's finite-element energy."""
        qF = self._qFE
        nC = self.mesh.nCells
        e, _ = self._energy(qF, qF.interpLocal @ T)
        energy = np.bincount(qF.cell, weights=qF.weight * e, minlength=nC)
        vol = np.bincount(qF.cell, weights=qF.weight, minlength=nC)
        out = np.full(nC, np.nan)
        for l, mat in enumerate(self.materials):
            cells = self.cells[self.cellLayer[self.cells] == l]
            if cells.size:
                out[cells] = mat.inverseEnergy(energy[cells] / (mat.density * vol[cells]))
        return out

    # ------------------------------------------------------------------ output
    def writeSeries(self, result: LayeredMapResult, path: str, name: str = "temperature") -> list:
        """Write one .vtu per time and a .pvd collection; returns the file list."""
        import os
        from pythonLibs.fieldMapping.io.Vtu import writePvd, writeVtu
        base, _ = os.path.splitext(path)
        files = []
        for k, t in enumerate(result.times):
            m = UnstructuredMesh(self.mesh.points, self.mesh.cellTypes, self.mesh.connectivity, self.mesh.offsets,
                                 self.mesh.polyFaces, dict(self.mesh.pointData), dict(self.mesh.cellData))
            m.pointData[name] = result.temperature[k]
            if result.cellTemperature is not None:
                m.cellData[name] = result.cellTemperature[k]
            m.cellData["conservationGroup"] = self.cellGroup
            f = f"{base}_{k:04d}.vtu"
            writeVtu(f, m)
            files.append((float(t), f))
        writePvd(path, files)
        return [f for _, f in files]
