"""FieldMappingHandler: toolBase service for conservative field transfer between meshes.

Meshes are loaded from .vtu files into named slots; commands map surface
fluxes (conserving incoming and outgoing heat) and layered 1-D profiles
(conserving thermal energy) and write .vtu / .pvd results::

    fm = FieldMappingHandler()
    fm.invoke("loadMesh", filePath="flow.vtu", meshName="flow")
    fm.invoke("loadMesh", filePath="structure.vtu", meshName="structure")
    fm.invoke("mapSurfaceFlux", sourceMesh="flow", targetMesh="structure", arrayName="q",
              outputFile="mappedFlux.vtu")
    fm.invoke("mapLayeredProfiles", targetMesh="structure", stationsFile="stations.json",
              materials="materials.json", outputFile="T.pvd")

All results are JSON-compatible dicts with the conservation diagnostics.

Note: annotations are evaluated at definition time on purpose (no
``from __future__ import annotations``) because ``buildCatalog`` derives
parameter types from them.
"""
import json
import os
from typing import Any, Optional

import numpy as np

from pythonLibs.tool.decorators import secure_expose
from pythonLibs.tool.toolBaseSecured import toolBaseSecured

from pythonLibs.fieldMapping.LayeredProfileMapper import LayeredProfileMapper
from pythonLibs.fieldMapping.StationIO import loadMaterials, loadStations
from pythonLibs.fieldMapping.SurfaceFluxMapper import SurfaceFluxMapper
from pythonLibs.fieldMapping.io.Vtu import readPvd, readVtu, writePvd, writeVtu
from pythonLibs.fieldMapping.mesh.Elements import TYPE_NAMES
from pythonLibs.fieldMapping.mesh.Mesh import UnstructuredMesh

_CATEGORY = "fieldMapping"


def _json(value):
    """Structured parameters may arrive as JSON text (CLI / REST)."""
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        return json.loads(value)
    return value


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _labelKey(k):
    """Material keys from JSON are strings; layer labels in files are numbers."""
    try:
        f = float(k)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return k


class FieldMappingHandler(toolBaseSecured):
    """Conservative field transfer service (surface fluxes, layered 1-D profiles) on .vtu meshes."""

    LAB_AUTOREGISTER = True
    LAB_NAME = "fieldMapping"
    service_name = "fieldMapping"

    parameterMap = {
        "filePath": "Path of a .vtu file",
        "meshName": "Name under which a mesh is stored",
        "sourceMesh": "Mesh carrying the source field",
        "targetMesh": "Mesh receiving the field (a volume mesh uses its boundary surface)",
        "arrayName": "Name of the source cell or point array",
        "location": "'cell' or 'point' (default: where the array is found)",
        "reconstruction": "Cell data: 'constant' or 'linear' (limited, conservative)",
        "pointFlux": "Also compute conservative nodal flux values",
        "groupArray": "Target cell array whose labels each conserve heat separately (nodal flux)",
        "maxDistance": "Largest gap between the surfaces",
        "maxAngle": "Largest angle (degrees) between matched facet normals",
        "orientation": "Source normal orientation: auto, same or opposite",
        "unmatched": "Heat without overlap: 'nearest' facet within maxDistance, or 'drop'",
        "outputFile": "Result file (.vtu, or .pvd for time series)",
        "sourceArea": "Source face areas: auto, vector (finite-volume), fan, fe, or a cell array with the solver's areas",
        "targetArea": "Target face areas for flux = heat / area: fe (default), vector, fan or a cell array",
        "nodalWeighting": "Nodal flux: 'area' (tributary face area) or 'consistent' (finite-element mass)",
        "sourceSeries": "Optional .pvd time series of the source (same geometry at every time)",
        "stations": "List of stations {position, time, depth, temperature, interfaces, name}",
        "stationsFile": "JSON station manifest (inline data or CSV temperature tables, see StationIO)",
        "materials": "{layer label: {density, cp (number or [[T, cp], ...])}} of the 3-D layers, or a JSON file",
        "layerArray": "Cell array with the layer labels",
        "layers": "Layer labels from the outer surface inward",
        "weights": "Station interpolation: kriging (default), idw or nearest",
        "lengthscale": "Kriging correlation length ('auto': maximum likelihood on the stations)",
        "depthMode": "surfaceAnchored (default), layerNormalized or depth",
        "conservation": "Energy groups: stationLayer, layer, global or cell",
        "times": "Output times (default: all station times)",
        "fmt": "VTU format: appended, binary or ascii",
        "kind": "Surface to extract: 'boundary' or 'interface'",
    }

    def __init__(self, variablesDict=None, caseName=None, *, secure_enabled: bool = False,
                 auth_handler=None, token=None, exposure_mode: str = "expose_only", workdir_root=None) -> None:
        self._meshes: dict = {}
        super().__init__(instance_subcls=None, variablesDict=variablesDict, caseName=caseName,
                         secure_enabled=secure_enabled, auth_handler=auth_handler, token=token,
                         exposure_mode=exposure_mode, workdir_root=workdir_root)

    # ------------------------------------------------------------------ Python accessors
    def getMesh(self, meshName: str) -> UnstructuredMesh:
        """Mesh loaded or stored under ``meshName`` (KeyError lists the known names)."""
        try:
            return self._meshes[meshName]
        except KeyError:
            raise KeyError(f"unknown mesh {meshName!r}; available: {sorted(self._meshes)}") from None

    def addMesh(self, meshName: str, mesh: UnstructuredMesh) -> None:
        """Store an in-memory mesh under ``meshName`` so the commands can use it without a file."""
        self._meshes[meshName] = mesh

    @staticmethod
    def _summary(mesh: UnstructuredMesh) -> dict:
        types, counts = np.unique(mesh.cellTypes, return_counts=True)
        dims = mesh.cellDimensions()
        lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
        return _plain({"nPoints": mesh.nPoints, "nCells": mesh.nCells,
                       "cellTypes": {TYPE_NAMES.get(int(t), str(int(t))): int(c) for t, c in zip(types, counts)},
                       "dimension": int(dims.max()) if dims.size else 0, "bounds": [lo, hi],
                       "pointData": {k: list(np.shape(v)) for k, v in mesh.pointData.items()},
                       "cellData": {k: list(np.shape(v)) for k, v in mesh.cellData.items()},
                       "measure": float(mesh.cellMeasures()[dims == dims.max()].sum()) if dims.size else 0.0})

    # ------------------------------------------------------------------ meshes
    @secure_expose(alias="loadMesh", category=_CATEGORY)
    def loadMesh(self, filePath: str, meshName: str) -> dict:
        """Load a .vtu file (any cell types, polyhedra included)."""
        mesh = readVtu(filePath)
        self._meshes[meshName] = mesh
        self.addExecutionInputFiles(file_paths=[filePath])
        return {"meshName": meshName, **self._summary(mesh)}

    @secure_expose(alias="listMeshes", category=_CATEGORY)
    def listMeshes(self) -> dict:
        """Loaded meshes."""
        return {"meshes": [{"meshName": k, "nPoints": v.nPoints, "nCells": v.nCells}
                           for k, v in sorted(self._meshes.items())]}

    @secure_expose(alias="meshSummary", category=_CATEGORY)
    def meshSummary(self, meshName: str) -> dict:
        """Cell types, bounds, arrays and total measure of a mesh."""
        return {"meshName": meshName, **self._summary(self.getMesh(meshName))}

    @secure_expose(alias="deleteMesh", category=_CATEGORY)
    def deleteMesh(self, meshName: str) -> dict:
        """Remove a mesh."""
        self.getMesh(meshName)
        del self._meshes[meshName]
        return {"deleted": meshName}

    @secure_expose(alias="writeMesh", category=_CATEGORY)
    def writeMesh(self, meshName: str, filePath: str, fmt: str = "appended") -> dict:
        """Write a mesh with its arrays to .vtu."""
        writeVtu(filePath, self.getMesh(meshName), fmt)
        self.addExecutionOutputFiles(file_paths=[filePath])
        return {"meshName": meshName, "filePath": filePath}

    @secure_expose(alias="extractSurface", category=_CATEGORY)
    def extractSurface(self, meshName: str, surfaceName: str, kind: str = "boundary",
                       layerArray: Optional[str] = None) -> dict:
        """Store the boundary surface (or the interfaces between the labels of layerArray) of a volume mesh."""
        mesh = self.getMesh(meshName)
        if kind == "boundary":
            surf = mesh.boundarySurface()
        elif kind == "interface":
            if not layerArray:
                raise ValueError("kind='interface' needs layerArray")
            surf = mesh.interfaceSurface(mesh.cellData[layerArray])
        else:
            raise ValueError("kind must be 'boundary' or 'interface'")
        self._meshes[surfaceName] = surf
        return {"surfaceName": surfaceName, **self._summary(surf)}

    # ------------------------------------------------------------------ surface flux
    @secure_expose(alias="mapSurfaceFlux", category=_CATEGORY)
    def mapSurfaceFlux(self, sourceMesh: str, targetMesh: str, arrayName: str, location: Optional[str] = None,
                       reconstruction: str = "constant", pointFlux: bool = False, groupArray: Optional[str] = None,
                       maxDistance: Optional[float] = None, maxAngle: float = 60.0, orientation: str = "auto",
                       unmatched: str = "nearest", resultName: Optional[str] = None,
                       outputFile: Optional[str] = None, sourceSeries: Optional[str] = None,
                       sourceArea: str = "auto", targetArea: str = "fe", nodalWeighting: str = "area") -> dict:
        """Conservative, sign-preserving transfer of a surface flux (incoming / outgoing heat conserved)."""
        src, tgt = self.getMesh(sourceMesh), self.getMesh(targetMesh)
        mapper = SurfaceFluxMapper(src, tgt, maxDistance=maxDistance, maxAngle=maxAngle, orientation=orientation,
                                   unmatched=unmatched, sourceArea=sourceArea, targetArea=targetArea,
                                   nodalWeighting=nodalWeighting)
        surf = mapper.target
        groups = None if groupArray is None else np.asarray(surf.cellData[groupArray])

        def locate(mesh):
            loc = location or ("cell" if arrayName in mesh.cellData else "point")
            data = mesh.cellData if loc == "cell" else mesh.pointData
            if arrayName not in data:
                raise KeyError(f"{loc} array {arrayName!r} not found")
            return np.asarray(data[arrayName], dtype=float), loc

        frames = [(0.0, src)] if sourceSeries is None else \
            [(d["time"], readVtu(d["file"])) for d in readPvd(sourceSeries)]
        name = resultName or arrayName
        outputs, diags = [], []
        for k, (t, mesh) in enumerate(frames):
            vals, loc = locate(mesh)
            res = mapper.map(vals, loc, reconstruction, pointFlux, groups)
            diags.append({"time": t, **res.diagnostics})
            out = UnstructuredMesh(surf.points, surf.cellTypes, surf.connectivity, surf.offsets, surf.polyFaces,
                                   {}, dict(surf.cellData))
            res.attach(out, name)
            if outputFile:
                if sourceSeries is None:
                    path = outputFile
                else:
                    base, _ = os.path.splitext(outputFile)
                    path = f"{base}_{k:04d}.vtu"
                writeVtu(path, out)
                outputs.append((t, path))
            self._meshes[f"{targetMesh}:{name}" if sourceSeries is None else f"{targetMesh}:{name}:{k}"] = out
        if outputFile and sourceSeries is not None:
            writePvd(outputFile, outputs)
        if outputFile:
            self.addExecutionOutputFiles(file_paths=[p for _, p in outputs] +
                                         ([outputFile] if sourceSeries is not None else []))
        return _plain({"targetMesh": targetMesh, "resultMesh": f"{targetMesh}:{name}", "arrayName": name,
                       "orientationSign": mapper.orientationSign, "nFrames": len(frames),
                       "outputFile": outputFile, "diagnostics": diags if len(diags) > 1 else diags[0]})

    # ------------------------------------------------------------------ layered profiles
    @secure_expose(alias="mapLayeredProfiles", category=_CATEGORY)
    def mapLayeredProfiles(self, targetMesh: str, materials: Any, stations: Optional[Any] = None,
                           stationsFile: Optional[str] = None, layerArray: str = "layer",
                           layers: Optional[list] = None, weights: str = "kriging", lengthscale: Any = "auto",
                           depthMode: str = "surfaceAnchored", conservation: str = "stationLayer", times: Optional[list] = None,
                           outputFile: Optional[str] = None, order: int = 4, referenceOrder: int = 10) -> dict:
        """Map layered 1-D temperature profiles onto a 3-D mesh with exact thermal energy conservation."""
        mesh = self.getMesh(targetMesh)
        if stationsFile:
            stations = loadStations(stationsFile)
            self.addExecutionInputFiles(file_paths=[stationsFile])
        stations = _json(stations)
        if not stations:
            raise ValueError("give stations or stationsFile")
        if isinstance(materials, str) and os.path.isfile(materials):
            self.addExecutionInputFiles(file_paths=[materials])
            materials = loadMaterials(materials)
        mats = {_labelKey(k): v for k, v in _json(materials).items()}
        if isinstance(lengthscale, str) and lengthscale != "auto":
            lengthscale = float(lengthscale)
        lay = [_labelKey(v) for v in _json(layers)] if layers is not None else None
        mapper = LayeredProfileMapper(mesh, stations, mats, layerArray=layerArray, layers=lay, weights=weights,
                                      lengthscale=lengthscale, depthMode=depthMode, conservation=conservation, order=order,
                                      referenceOrder=referenceOrder)
        res = mapper.map(_json(times))
        files = []
        if outputFile:
            files = mapper.writeSeries(res, outputFile)
            self.addExecutionOutputFiles(file_paths=files + [outputFile])
        last = UnstructuredMesh(mesh.points, mesh.cellTypes, mesh.connectivity, mesh.offsets, mesh.polyFaces,
                                dict(mesh.pointData), dict(mesh.cellData))
        last.pointData["temperature"] = res.temperature[-1]
        if res.cellTemperature is not None:
            last.cellData["temperature"] = res.cellTemperature[-1]
        self._meshes[f"{targetMesh}:temperature"] = last
        compact = [{k: d[k] for k in ("time", "converged", "iterations", "maxRelativeEnergyError",
                                      "maxRelativeNodalEnergyError", "maxCorrection", "nodesAtBounds")}
                   for d in res.diagnostics]
        return _plain({"targetMesh": targetMesh, "resultMesh": f"{targetMesh}:temperature",
                       "nGroups": mapper.nGroups, "lengthscale": mapper.lengthscale, "nStations": len(mapper.stations), "outputFile": outputFile,
                       "summary": res.summary(), "steps": compact})
