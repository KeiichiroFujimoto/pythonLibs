"""Reading 1-D through-thickness station results and material tables from files.

Stations are listed in a JSON manifest; each entry gives its position, its layer
interface depths and its temperature history, inline or as CSV files::

    {"stations": [
       {"name": "S1", "position": [0.0, 0.0, 1.0],
        "interfaces": [0.0, 0.020, 0.025],          # or "interfacesFile": "S1_interfaces.csv"
        "file": "S1.csv"},                           # or inline "time", "depth", "temperature"
       ...]}

A plain JSON list of station dicts is accepted as well. Paths are relative to the manifest.

Temperature CSV (wide format): the header row holds the depths below the current surface,
every following row a time and the temperatures at those depths::

    time, 0.0, 0.0005, 0.001, ...
    0.0,  300, 300,    300,   ...
    10.0, 1450, 1320,  1190,  ...

When the depth grid moves with time (recession), give ``"depthFile"`` with the same layout
(time, depths of every column); the temperature header then holds column labels (T1, T2, ...).
Interface CSV: rows of time, b0 (= 0), b1, ..., bL. Lines starting with '#' are comments;
commas, semicolons, tabs or spaces separate the values.

Materials: JSON {"<layer label>": {"density": rho, "cp": number or [[T, cp], ...], "name": ...}}.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np

_SPLIT = re.compile(r"[,;\t ]+")


def readTable(path: str) -> tuple[list[str], np.ndarray]:
    """(header entries, numeric rows) of a delimited text file; '#' starts a comment line."""
    header, rows = None, []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            values = [v for v in _SPLIT.split(text) if v != ""]
            if header is None:
                try:
                    rows.append([float(v) for v in values])
                    header = []
                except ValueError:
                    header = values
                continue
            rows.append([float(v) for v in values])
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    width = {len(r) for r in rows}
    if len(width) != 1:
        raise ValueError(f"{path}: rows have different lengths {sorted(width)}")
    return header or [], np.array(rows, dtype=float)


def readProfileCsv(path: str, depthFile: Optional[str] = None) -> dict:
    """{"time", "depth", "temperature"} from a wide temperature table (see the module docstring)."""
    header, data = readTable(path)
    time, temp = data[:, 0], data[:, 1:]
    if depthFile:
        _, d = readTable(depthFile)
        if d.shape != data.shape or not np.allclose(d[:, 0], time):
            raise ValueError(f"{depthFile}: must match the times and columns of {path}")
        depth = d[:, 1:]
    else:
        try:
            depth = np.array([float(v) for v in header[1:]])
        except ValueError:
            raise ValueError(f"{path}: the header must list the depths (or give depthFile)") from None
        if depth.size != temp.shape[1]:
            raise ValueError(f"{path}: {depth.size} depths in the header, {temp.shape[1]} temperature columns")
    return {"time": time, "depth": depth, "temperature": temp}


def loadStations(path: str) -> list[dict]:
    """Station specs from a JSON manifest (inline data or CSV files), ready for LayeredProfileMapper."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entries = raw["stations"] if isinstance(raw, dict) else raw
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for i, e in enumerate(entries):
        st = {"name": e.get("name", f"S{i + 1}"), "position": e["position"]}
        if "file" in e:
            depthFile = os.path.join(base, e["depthFile"]) if e.get("depthFile") else None
            st.update(readProfileCsv(os.path.join(base, e["file"]), depthFile))
        else:
            st.update({"time": e["time"], "depth": e["depth"], "temperature": e["temperature"]})
        if "interfacesFile" in e:
            _, b = readTable(os.path.join(base, e["interfacesFile"]))
            if not np.allclose(b[:, 0], np.asarray(st["time"], dtype=float)):
                raise ValueError(f"{e['interfacesFile']}: times differ from the temperature table")
            st["interfaces"] = b[:, 1:]
        else:
            st["interfaces"] = e["interfaces"]
        out.append(st)
    return out


def loadMaterials(path: str) -> dict:
    """Materials per layer label from JSON (labels converted to numbers where possible)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {_label(k): v for k, v in raw.items()}


def _label(k):
    try:
        f = float(k)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return k


def writeStationTemplate(directory: str, name: str = "S1") -> str:
    """Write an example manifest + CSV pair (a starting point for real data); returns the manifest path."""
    os.makedirs(directory, exist_ok=True)
    depth = np.linspace(0.0, 0.025, 11)
    times = np.array([0.0, 30.0, 60.0])
    with open(os.path.join(directory, f"{name}.csv"), "w", encoding="utf-8") as f:
        f.write("# time [s], then temperatures [K] at the depths [m] of the header\n")
        f.write("time," + ",".join(f"{d:g}" for d in depth) + "\n")
        for t in times:
            T = 300.0 + (t / 60.0) * 1500.0 * np.exp(-depth / 0.004)
            f.write(f"{t:g}," + ",".join(f"{v:.3f}" for v in T) + "\n")
    manifest = os.path.join(directory, "stations.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"stations": [{"name": name, "position": [0.0, 0.0, 1.0], "interfaces": [0.0, 0.02, 0.025],
                                 "file": f"{name}.csv"}]}, f, indent=2)
    return manifest
