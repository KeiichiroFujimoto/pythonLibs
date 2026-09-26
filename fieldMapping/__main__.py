"""Command line: python -m pythonLibs.fieldMapping <command> ...

    info      mesh.vtu                               cell types, bounds, arrays, measure
    template  DIR                                    example station manifest + CSV + materials
    layered   --mesh M.vtu --stations S.json --materials MAT.json --out T.pvd
              [--layer-array layer] [--layers 1 2] [--times 0 30 60] [--weights kriging]
              [--lengthscale auto] [--depth-mode surfaceAnchored] [--conservation stationLayer]
    flux      --source flow.vtu --target structure.vtu --array q --out mapped.vtu
              [--location cell] [--source-series flow.pvd] [--source-area auto] [--target-area fe]
              [--point-flux] [--group-array ARR] [--max-distance D] [--max-angle 60]

Every command prints a JSON summary (conservation diagnostics included); --report FILE
also writes it to a file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pythonLibs.fieldMapping.FieldMappingHandler import FieldMappingHandler
from pythonLibs.fieldMapping.StationIO import writeStationTemplate


def _labels(values):
    if values is None:
        return None
    out = []
    for v in values:
        try:
            f = float(v)
            out.append(int(f) if f.is_integer() else f)
        except ValueError:
            out.append(v)
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pythonLibs.fieldMapping",
                                description="Conservative mapping of thermal data between meshes (VTU).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="summary of a .vtu mesh")
    s.add_argument("mesh")
    s.add_argument("--report")

    s = sub.add_parser("template", help="write example station / material input files")
    s.add_argument("directory")
    s.add_argument("--report")

    s = sub.add_parser("layered", help="map layered 1-D temperature profiles onto a 3-D mesh (energy conserved)")
    s.add_argument("--mesh", required=True)
    s.add_argument("--stations", required=True, help="station manifest (JSON)")
    s.add_argument("--materials", required=True, help="materials per layer label (JSON)")
    s.add_argument("--out", help="output .pvd (one .vtu per time) or .vtu")
    s.add_argument("--layer-array", default="layer")
    s.add_argument("--layers", nargs="+", help="layer labels from the outer surface inward")
    s.add_argument("--times", nargs="+", type=float)
    s.add_argument("--weights", default="kriging", choices=("kriging", "idw", "nearest"))
    s.add_argument("--lengthscale", default="auto")
    s.add_argument("--depth-mode", default="surfaceAnchored", choices=("surfaceAnchored", "layerNormalized", "depth"))
    s.add_argument("--conservation", default="stationLayer", choices=("stationLayer", "layer", "global", "cell"))
    s.add_argument("--order", type=int, default=4)
    s.add_argument("--report")

    s = sub.add_parser("flux", help="map a surface heat flux between meshes (incoming / outgoing heat conserved)")
    s.add_argument("--source", required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--array", required=True)
    s.add_argument("--out")
    s.add_argument("--location", choices=("cell", "point"))
    s.add_argument("--source-series", help=".pvd time series of the source")
    s.add_argument("--source-area", default="auto")
    s.add_argument("--target-area", default="fe")
    s.add_argument("--nodal-weighting", default="area", choices=("area", "consistent"))
    s.add_argument("--reconstruction", default="constant", choices=("constant", "linear"))
    s.add_argument("--point-flux", action="store_true")
    s.add_argument("--group-array")
    s.add_argument("--max-distance", type=float)
    s.add_argument("--max-angle", type=float, default=60.0)
    s.add_argument("--orientation", default="auto", choices=("auto", "same", "opposite"))
    s.add_argument("--report")
    return p


def run(argv=None) -> dict:
    """Execute one command-line command (``info``, ``template``, ``layered`` or ``flux``) and return its result."""
    a = _parser().parse_args(argv)
    fm = FieldMappingHandler()
    if a.command == "info":
        out = fm.loadMesh(a.mesh, os.path.basename(a.mesh))
    elif a.command == "template":
        manifest = writeStationTemplate(a.directory)
        materials = os.path.join(a.directory, "materials.json")
        with open(materials, "w", encoding="utf-8") as f:
            json.dump({"1": {"name": "ablator", "density": 280.0, "cp": [[300, 1000], [2000, 1800]]},
                       "2": {"name": "structure", "density": 2700.0, "cp": [[300, 900], [800, 1100]]}}, f, indent=2)
        out = {"stations": manifest, "materials": materials}
    elif a.command == "layered":
        fm.loadMesh(a.mesh, "target")
        out = fm.mapLayeredProfiles("target", a.materials, stationsFile=a.stations, layerArray=a.layer_array,
                                    layers=_labels(a.layers), weights=a.weights, lengthscale=a.lengthscale,
                                    depthMode=a.depth_mode, conservation=a.conservation, times=a.times,
                                    outputFile=a.out, order=a.order)
    else:
        fm.loadMesh(a.source, "source")
        fm.loadMesh(a.target, "target")
        out = fm.mapSurfaceFlux("source", "target", a.array, location=a.location,
                                reconstruction=a.reconstruction, pointFlux=a.point_flux, groupArray=a.group_array,
                                maxDistance=a.max_distance, maxAngle=a.max_angle, orientation=a.orientation,
                                outputFile=a.out, sourceSeries=a.source_series, sourceArea=a.source_area,
                                targetArea=a.target_area, nodalWeighting=a.nodal_weighting)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def main(argv=None) -> int:
    """Command-line entry point: print the result as JSON; exit code 1 on input errors."""
    try:
        out = run(argv)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
