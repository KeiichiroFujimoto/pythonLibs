"""VTK XML unstructured grid (.vtu) and collection (.pvd) reader / writer, numpy + standard library only.

Reading supports every encoding written by VTK: ``ascii``, inline ``binary``
(base64) and ``appended`` data (``raw`` or ``base64``), uncompressed or
compressed (vtkZLibDataCompressor, vtkLZMADataCompressor), UInt32 / UInt64
headers and either byte order, several Pieces (merged), and arbitrary
polyhedra in both the classic ``faces`` / ``faceoffsets`` layout and the
newer face-connectivity layout. Writing produces ``appended`` raw (default,
zlib-compressed, UInt64 headers), inline ``binary`` or ``ascii`` files.

Field arrays keep their names and component counts; one-component arrays are
returned as (n,), others as (n, k).
"""
from __future__ import annotations

import base64
import lzma
import os
import re
import xml.etree.ElementTree as ET
import zlib
from typing import Optional

import numpy as np

from pythonLibs.fieldMapping.mesh.Mesh import UnstructuredMesh

_DTYPES = {"Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2", "Int32": "i4", "UInt32": "u4",
           "Int64": "i8", "UInt64": "u8", "Float32": "f4", "Float64": "f8"}
_NAMES = {np.dtype(v).str[1:]: k for k, v in _DTYPES.items()}


# ---------------------------------------------------------------- decoding
class _Decoder:
    def __init__(self, root: ET.Element, appended: Optional[bytes], appendedEncoding: str) -> None:
        self.order = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
        self.header = np.dtype(self.order + _DTYPES[root.get("header_type", "UInt32")])
        comp = root.get("compressor")
        if comp in (None, ""):
            self.decompress = None
        elif comp in ("vtkZLibDataCompressor", "vtkZLibDataCompression"):
            self.decompress = zlib.decompress
        elif comp in ("vtkLZMADataCompressor", "vtkLZMADataCompression"):
            self.decompress = lzma.decompress
        else:
            raise ValueError(f"unsupported compressor {comp!r} (zlib and lzma are supported)")
        self.appended = appended
        self.appendedEncoding = appendedEncoding
        # base64 appended arrays run from their offset to the next array's offset
        offs = sorted({int(a.get("offset")) for a in root.iter("DataArray") if a.get("format") == "appended"})
        end = len(appended.rstrip().rsplit(b"</AppendedData>", 1)[0].rstrip()) if appended is not None else 0
        self._next = {o: n for o, n in zip(offs, offs[1:] + [end])}

    def array(self, node: ET.Element) -> np.ndarray:
        dtype = np.dtype(self.order + _DTYPES[node.get("type")])
        ncomp = int(node.get("NumberOfComponents", "1"))
        fmt = node.get("format", "ascii")
        if fmt == "ascii":
            text = (node.text or "").split()
            data = np.array(text, dtype=float if dtype.kind == "f" else np.int64).astype(dtype.newbyteorder("="))
        elif fmt == "binary":
            data = self._binary((node.text or "").strip().encode("ascii"), dtype, encoded=True)
        elif fmt == "appended":
            if self.appended is None:
                raise ValueError("appended DataArray without AppendedData")
            off = int(node.get("offset"))
            if self.appendedEncoding == "raw":
                data = self._raw(self.appended, off, dtype)
            else:
                data = self._binary(self.appended[off:self._next[off]].strip(), dtype, encoded=True)
        else:
            raise ValueError(f"unknown DataArray format {fmt!r}")
        data = data.astype(dtype.newbyteorder("="), copy=False)
        return data.reshape(-1, ncomp) if ncomp > 1 else data

    def _decompress(self, header: np.ndarray, blob: bytes, dtype) -> np.ndarray:
        nb = int(header[0])
        sizes = header[3:3 + nb].astype(np.int64)
        out, pos = [], 0
        for s in sizes:
            out.append(self.decompress(blob[pos:pos + int(s)]))
            pos += int(s)
        return np.frombuffer(b"".join(out), dtype=dtype)

    def _raw(self, buf: bytes, off: int, dtype) -> np.ndarray:
        hs = self.header.itemsize
        if self.decompress is None:
            n = int(np.frombuffer(buf, self.header, 1, off)[0])
            return np.frombuffer(buf, dtype, n // dtype.itemsize, off + hs)
        nb = int(np.frombuffer(buf, self.header, 1, off)[0])
        header = np.frombuffer(buf, self.header, 3 + nb, off)
        start = off + (3 + nb) * hs
        total = int(np.sum(header[3:3 + nb]))
        return self._decompress(header, buf[start:start + total], dtype)

    def _binary(self, text: bytes, dtype, encoded: bool) -> np.ndarray:
        hs = self.header.itemsize
        text = re.sub(rb"\s+", b"", text)
        if self.decompress is None:
            raw = base64.b64decode(text)
            n = int(np.frombuffer(raw[:hs], self.header, 1)[0])
            if len(raw) >= hs + n:
                return np.frombuffer(raw, dtype, n // dtype.itemsize, hs)
            # header encoded separately from the data
            hl = 4 * ((hs + 2) // 3)
            n = int(np.frombuffer(base64.b64decode(text[:hl])[:hs], self.header, 1)[0])
            return np.frombuffer(base64.b64decode(text[hl:]), dtype, n // dtype.itemsize)
        # compressed: the header (3 + nBlocks integers) is base64-encoded on its own
        first = base64.b64decode(text[:4 * ((hs + 2) // 3) + 8])
        nb = int(np.frombuffer(first[:hs], self.header, 1)[0])
        hl = 4 * (((3 + nb) * hs + 2) // 3)
        header = np.frombuffer(base64.b64decode(text[:hl]), self.header, 3 + nb)
        return self._decompress(header, base64.b64decode(text[hl:]), dtype)


def _split(raw: bytes) -> tuple[bytes, Optional[bytes], str]:
    """(XML part, appended bytes after '_', encoding)."""
    i = raw.find(b"<AppendedData")
    if i < 0:
        return raw, None, "raw"
    tagEnd = raw.find(b">", i)
    tag = raw[i:tagEnd + 1].decode("ascii")
    enc = re.search(r'encoding="(\w+)"', tag)
    under = raw.find(b"_", tagEnd)
    head = raw[:i] + b"</VTKFile>"
    data = raw[under + 1:]
    return head, data, enc.group(1) if enc else "raw"


# ---------------------------------------------------------------- reading
def readVtu(path: str) -> UnstructuredMesh:
    """Read a .vtu file into an UnstructuredMesh (all Pieces merged)."""
    with open(path, "rb") as f:
        raw = f.read()
    head, appended, enc = _split(raw)
    root = ET.fromstring(head)
    if root.get("type") != "UnstructuredGrid":
        raise ValueError(f"{path}: not an UnstructuredGrid file (type={root.get('type')!r})")
    dec = _Decoder(root, appended, enc)
    grid = root.find("UnstructuredGrid")
    meshes = [_readPiece(p, dec) for p in grid.findall("Piece")]
    if not meshes:
        raise ValueError(f"{path}: no Piece")
    return meshes[0] if len(meshes) == 1 else UnstructuredMesh.merge(meshes)


def _arrays(node: Optional[ET.Element], dec: _Decoder) -> dict:
    if node is None:
        return {}
    return {a.get("Name", f"array{i}"): dec.array(a) for i, a in enumerate(node.findall("DataArray"))}


def _readPiece(piece: ET.Element, dec: _Decoder) -> UnstructuredMesh:
    npts = int(piece.get("NumberOfPoints"))
    ncells = int(piece.get("NumberOfCells"))
    pts = dec.array(piece.find("Points").find("DataArray")).astype(float).reshape(npts, -1)
    if pts.shape[1] < 3:
        pts = np.hstack([pts, np.zeros((npts, 3 - pts.shape[1]))])
    cells = {a.get("Name"): a for a in piece.find("Cells").findall("DataArray")}
    conn = dec.array(cells["connectivity"]).astype(np.int64)
    offs = dec.array(cells["offsets"]).astype(np.int64)
    types = dec.array(cells["types"]).astype(np.uint8)
    if offs.size != ncells or types.size != ncells:
        raise ValueError("cell arrays do not match NumberOfCells")
    polyFaces = None
    if "faces" in cells and "faceoffsets" in cells:
        polyFaces = _classicFaces(dec.array(cells["faces"]).astype(np.int64),
                                  dec.array(cells["faceoffsets"]).astype(np.int64), types)
    elif "face_connectivity" in cells:
        polyFaces = _newFaces({k: dec.array(v).astype(np.int64) for k, v in cells.items()}, types)
    mesh = UnstructuredMesh(pts, types, conn, np.concatenate([[0], offs]), polyFaces=polyFaces)
    mesh.pointData.update(_arrays(piece.find("PointData"), dec))
    mesh.cellData.update(_arrays(piece.find("CellData"), dec))
    return mesh


def _classicFaces(faces, faceOffsets, types) -> dict:
    """{cell: [face node arrays]} from the classic faces / faceoffsets stream."""
    out, start = {}, 0
    for c in np.flatnonzero(types == 42):
        end = int(faceOffsets[c])                        # each cell's stream starts at the previous end
        stream = faces[start:end]
        start = end
        nf, pos, fl = int(stream[0]), 1, []
        for _ in range(nf):
            k = int(stream[pos])
            fl.append(stream[pos + 1:pos + 1 + k].copy())
            pos += 1 + k
        out[int(c)] = fl
    return out


def _newFaces(arr: dict, types) -> dict:
    """{cell: [face node arrays]} from the face_connectivity / polyhedron_to_faces layout (VTK >= 9.4).

    Offsets are cumulative ends (a leading 0 is accepted); polyhedron_offsets
    may have one entry per cell (as VTK writes) or one per polyhedral cell.
    """
    fc, fo = arr["face_connectivity"], arr["face_offsets"]
    pf, po = arr["polyhedron_to_faces"], arr["polyhedron_offsets"]
    fo = fo if fo.size and fo[0] == 0 else np.concatenate([[0], fo])
    polys = np.flatnonzero(types == 42)
    nCells = types.size

    def cumulative(a, n):
        if a.size == n + 1 and a[0] == 0:
            return a
        return np.concatenate([[0], a]) if a.size == n else None

    perCell = cumulative(po, nCells)
    if perCell is not None and perCell[-1] == pf.size and \
            np.all(np.diff(perCell)[types != 42] == 0):
        po, idx = perCell, {int(c): int(c) for c in polys}
    else:
        po = cumulative(po, polys.size)
        if po is None or po[-1] != pf.size:
            raise ValueError("inconsistent polyhedron_offsets")
        idx = {int(c): k for k, c in enumerate(polys)}
    out = {}
    for c in polys:
        k = idx[int(c)]
        out[int(c)] = [fc[fo[f]:fo[f + 1]].copy() for f in pf[po[k]:po[k + 1]]]
    return out


# ---------------------------------------------------------------- writing
def _encode(data: np.ndarray, compress: bool, header: np.dtype, level: int = 6) -> bytes:
    raw = np.ascontiguousarray(data).tobytes()
    if not compress:
        return np.array([len(raw)], dtype=header).tobytes() + raw
    block = 1 << 20
    chunks = [raw[i:i + block] for i in range(0, len(raw), block)] or [b""]
    comp = [zlib.compress(c, level) for c in chunks]
    last = len(chunks[-1]) if raw else 0
    head = np.array([len(chunks), block, last] + [len(c) for c in comp], dtype=header).tobytes()
    return head + b"".join(comp)


def _encodeB64(data: np.ndarray, compress: bool, header: np.dtype) -> bytes:
    blob = _encode(data, compress, header)
    if not compress:
        return base64.b64encode(blob)
    nb = int(np.frombuffer(blob, header, 1)[0])
    hl = (3 + nb) * header.itemsize
    return base64.b64encode(blob[:hl]) + base64.b64encode(blob[hl:])


def _typeName(a: np.ndarray) -> str:
    key = a.dtype.str[1:]
    if key not in _NAMES:
        a = a.astype(np.float64)
        key = "f8"
    return _NAMES[key]


def writeVtu(path: str, mesh: UnstructuredMesh, fmt: str = "appended", compress: bool = True) -> None:
    """Write ``mesh`` (points, cells, polyhedra, point / cell data) to a .vtu file.

    fmt: "appended" (raw binary, default), "binary" (inline base64) or "ascii".
    """
    if fmt not in ("appended", "binary", "ascii"):
        raise ValueError("fmt must be appended, binary or ascii")
    compress = compress and fmt != "ascii"
    header = np.dtype("<u8")
    arrays = []                                            # (section, name, array)
    arrays.append(("Points", "Points", mesh.points.astype(np.float64)))
    arrays.append(("Cells", "connectivity", mesh.connectivity.astype(np.int64)))
    arrays.append(("Cells", "offsets", mesh.offsets[1:].astype(np.int64)))
    arrays.append(("Cells", "types", mesh.cellTypes.astype(np.uint8)))
    if mesh.polyFaces:
        faces, fo, pos = [], np.full(mesh.nCells, -1, dtype=np.int64), 0
        for c in range(mesh.nCells):
            if c in mesh.polyFaces:
                fl = mesh.polyFaces[c]
                stream = [len(fl)] + [v for f in fl for v in [len(f), *map(int, f)]]
                faces.extend(stream)
                pos += len(stream)
                fo[c] = pos
        arrays.append(("Cells", "faces", np.array(faces, dtype=np.int64)))
        arrays.append(("Cells", "faceoffsets", fo))
    for name, a in mesh.pointData.items():
        arrays.append(("PointData", name, np.asarray(a)))
    for name, a in mesh.cellData.items():
        arrays.append(("CellData", name, np.asarray(a)))

    attrs = 'type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64"'
    if compress:
        attrs += ' compressor="vtkZLibDataCompressor"'
    lines = ['<?xml version="1.0"?>', f"<VTKFile {attrs}>", "<UnstructuredGrid>",
             f'<Piece NumberOfPoints="{mesh.nPoints}" NumberOfCells="{mesh.nCells}">']
    blobs, offset = [], 0

    def dataArray(name, a):
        nonlocal offset
        a = np.asarray(a)
        ncomp = 1 if a.ndim == 1 else a.shape[1]
        t = _typeName(a)
        a = a.astype(np.dtype("<" + _DTYPES[t]), copy=False)
        common = f'type="{t}" Name="{name}" NumberOfComponents="{ncomp}"'
        if fmt == "ascii":
            body = " ".join(repr(float(v)) if a.dtype.kind == "f" else str(int(v)) for v in a.ravel())
            return f'<DataArray {common} format="ascii">{body}</DataArray>'
        if fmt == "binary":
            return f'<DataArray {common} format="binary">{_encodeB64(a, compress, header).decode("ascii")}</DataArray>'
        blob = _encode(a, compress, header)
        tag = f'<DataArray {common} format="appended" offset="{offset}"/>'
        blobs.append(blob)
        offset += len(blob)
        return tag

    for section in ("PointData", "CellData"):
        items = [(n, a) for s, n, a in arrays if s == section]
        lines.append(f"<{section}>")
        lines.extend(dataArray(n, a) for n, a in items)
        lines.append(f"</{section}>")
    lines.append("<Points>")
    lines.append(dataArray("Points", arrays[0][2]))
    lines.append("</Points>")
    lines.append("<Cells>")
    lines.extend(dataArray(n, a) for s, n, a in arrays if s == "Cells")
    lines.append("</Cells>")
    lines.append("</Piece>")
    lines.append("</UnstructuredGrid>")
    with open(path, "wb") as f:
        f.write("\n".join(lines).encode("utf-8"))
        if fmt == "appended":
            f.write(b'\n<AppendedData encoding="raw">\n_')
            for b in blobs:
                f.write(b)
            f.write(b"\n</AppendedData>")
        f.write(b"\n</VTKFile>\n")


# ---------------------------------------------------------------- collections
def readPvd(path: str) -> list[dict]:
    """Datasets of a .pvd collection: [{"time": t, "file": absolute path, "part": p}, ...] sorted by time."""
    root = ET.parse(path).getroot()
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for ds in root.iter("DataSet"):
        out.append({"time": float(ds.get("timestep", "0")), "part": int(ds.get("part", "0")),
                    "file": os.path.join(base, ds.get("file"))})
    return sorted(out, key=lambda d: (d["time"], d["part"]))


def writePvd(path: str, entries: list[tuple[float, str]]) -> None:
    """Write a .pvd collection of (time, vtu file path) entries (paths stored relative to the .pvd)."""
    base = os.path.dirname(os.path.abspath(path))
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             "<Collection>"]
    for t, f in entries:
        rel = os.path.relpath(os.path.abspath(f), base)
        lines.append(f'<DataSet timestep="{float(t)!r}" group="" part="0" file="{rel}"/>')
    lines += ["</Collection>", "</VTKFile>"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
