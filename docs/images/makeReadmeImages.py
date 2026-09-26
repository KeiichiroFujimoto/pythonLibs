"""Generate the overview images used by the README files (light and dark variants).

Every number and curve in the images is computed here with the library itself, so the
images stay truthful when the code changes. Run from the directory that contains the
``pythonLibs`` package (or with the package installed):

    python -m pythonLibs.docs.images.makeReadmeImages

Output: ``docs/images/<name>-light.svg`` and ``<name>-dark.svg`` for toolBase,
regressionHandler and fieldMapping. The README files pick the variant matching the
GitHub theme with ``<picture>`` / ``prefers-color-scheme``.
"""
import os
from html import escape

import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WIDTH = 1000
SANS = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

THEMES = {
    "light": dict(surface="#fcfcfb", border="#e1e0d9", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", codeBg="#f3f2ee", accent="#2a78d6", accent2="#eb6834",
                  good="#006300", divNeg="#184f95", divMid="#f0efec", divPos="#b3282d"),
    "dark": dict(surface="#1a1a19", border="#2c2c2a", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", codeBg="#242423", accent="#3987e5", accent2="#d95926",
                 good="#0ca30c", divNeg="#3987e5", divMid="#383835", divPos="#e66767"),
}


def _text(x, y, s, size=14, fill="#000", weight="normal", family=SANS, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family=\'{family}\' font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" xml:space="preserve" {extra}>{escape(s)}</text>')


def _svg(height, t, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{escape(title)}">\n'
            f'<title>{escape(title)}</title>\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="12" fill="{t["surface"]}" '
            f'stroke="{t["border"]}"/>\n' + "\n".join(body) + "\n</svg>\n")


def _header(t, title, subtitle):
    return [_text(32, 50, title, 24, t["ink"], "600"), _text(32, 76, subtitle, 15, t["ink2"])]


def _write(name, mode, svg):
    path = os.path.join(OUT_DIR, f"{name}-{mode}.svg")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path


# ---------------------------------------------------------------------------
# toolBase: one decorated method -> six interfaces
# ---------------------------------------------------------------------------
def _toolBaseFacts():
    """Run the example service so the image shows real results."""
    from pythonLibs.tool.decorators import secure_expose
    from pythonLibs.tool.toolBaseSecured import toolBaseSecured

    class BeamService(toolBaseSecured):
        def __init__(self):
            super().__init__(instance_subcls=self, exposure_mode="expose_only", secure_enabled=False)

        @secure_expose(alias="tipDeflection", category="structures")
        def tip_deflection(self, load: float, length: float) -> dict:
            """Tip deflection of a cantilever.

            Args:
                load (float): End load [N].
                length (float): Beam length [m].
            """
            return {"deflectionM": load * length ** 3 / (3.0 * 200e9 * 8e-6)}

    svc = BeamService()
    result = svc.invoke("tipDeflection", load=1000.0, length=2.0)
    catalog = svc.buildCatalog()
    return {"deflection": result["deflectionM"], "nParams": len(catalog[0]["params"])}


def toolBaseImage(t, facts):
    body = _header(t, "toolBase — write a method once, use it everywhere",
                   "Decorate a method with @secure_expose; the same operation is exposed on six interfaces.")
    cx, cy, cw, ch = 32, 104, 376, 334
    body.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="8" fill="{t["codeBg"]}"/>')
    body.append(_text(cx + 18, cy + 28, "YOUR CODE", 11, t["muted"], "600", extra='letter-spacing="1.2"'))
    code = [
        ("class BeamService(toolBaseSecured):", "ink"),
        ("", "ink"),
        ('    @secure_expose(alias="tipDeflection",', "accent"),
        ('                   category="structures")', "accent"),
        ("    def tip_deflection(self, load: float,", "ink"),
        ("                       length: float) -> dict:", "ink"),
        ('        """Tip deflection of a cantilever.', "ink2"),
        ("", "ink"),
        ("        Args:", "ink2"),
        ("            load (float): End load [N].", "ink2"),
        ("            length (float): Beam length [m].", "ink2"),
        ('        """', "ink2"),
        ('        return {"deflectionM": ...}', "ink"),
    ]
    for i, (line, role) in enumerate(code):
        body.append(_text(cx + 18, cy + 60 + 20 * i, line, 12.5, t[role], family=MONO))

    rows = [
        ("Python call", 'svc.invoke("tipDeflection", load=1e3, length=2)',
         f'→ {{"deflectionM": {facts["deflection"]:.6f}}}'),
        ("Catalog · REST", "svc.buildCatalog()   GET /api/commands",
         f'→ {facts["nParams"]} typed params with their descriptions'),
        ("Interactive CLI", "svc.toCLI().run()", "service> tipDeflection load=1000 length=2"),
        ("LLM agent tools", "svc.toLangchainTools()", "→ StructuredTool + JSON schema of the arguments"),
        ("Parallel sweeps", 'svc.executeParallel("tipDeflection", cases)', "→ one result per case, thread pool"),
        ("Execution trace", "ToolTrace().attach(svc)", "→ jobs, events, durations, errors as JSON"),
    ]
    rx, rw, rh, gap = 462, 506, 50, 7
    hubX, hubY = cx + cw, cy + ch / 2
    for i, (name, call, out) in enumerate(rows):
        ry = cy + i * (rh + gap)
        mid = ry + rh / 2
        body.append(f'<path d="M{hubX},{hubY:.1f} C{hubX + 36},{hubY:.1f} {rx - 36},{mid:.1f} {rx - 6},{mid:.1f}" '
                    f'fill="none" stroke="{t["accent"]}" stroke-width="1.5" opacity="0.7"/>')
        body.append(f'<circle cx="{rx - 4}" cy="{mid:.1f}" r="3" fill="{t["accent"]}"/>')
        body.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="8" fill="none" stroke="{t["border"]}"/>')
        body.append(_text(rx + 14, ry + 20, name, 14, t["ink"], "600"))
        body.append(_text(rx + 146, ry + 20, call, 12, t["accent"], family=MONO))
        body.append(_text(rx + 146, ry + 39, out, 12, t["ink2"], family=MONO))
    body.append(_text(32, 474, "Also on the same base: optional access control with your own auth handler, unit-aware "
                      "arguments (@with_units), settings", 13, t["muted"]))
    body.append(_text(32, 494, "parameters, and the numpy-only regressionHandler and fieldMapping services.", 13,
                      t["muted"]))
    return _svg(516, t, body, "toolBase overview")


# ---------------------------------------------------------------------------
# regressionHandler: Kriging with a prediction band, real fit
# ---------------------------------------------------------------------------
def _regressionFacts():
    from pythonLibs.regressionHandler import createModel, crossValidate

    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.0, 10.0, 14))[:, None]
    truth = lambda v: np.sin(v) + 0.1 * v                                    # noqa: E731
    y = truth(x[:, 0]) + rng.normal(0.0, 0.08, x.shape[0])
    model = createModel("kriging").fit(x, y)
    xs = np.linspace(0.0, 10.0, 241)[:, None]
    band = model.predictInterval(xs, level=0.95, kind="prediction")
    rmse = float(crossValidate(model, x, y, method="analytic").rmse[0])     # exact leave-one-out
    return dict(x=x[:, 0], y=y, xs=xs[:, 0], mean=band.mean[:, 0], lo=band.lower[:, 0], hi=band.upper[:, 0],
                truth=truth(xs[:, 0]), looRmse=rmse)


def regressionImage(t, f):
    body = _header(t, "regressionHandler — surrogate models with uncertainty, numpy only",
                   "Fit, predict with intervals and gradients, validate, save as JSON: one API for 30+ model types.")
    px, py, pw, ph = 72, 126, 520, 290
    xmin, xmax = 0.0, 10.0
    ymin, ymax = -1.5, 2.5
    sx = lambda v: px + (v - xmin) / (xmax - xmin) * pw                      # noqa: E731
    sy = lambda v: py + ph - (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin) * ph  # noqa: E731
    for yv in (-1, 0, 1, 2):
        body.append(f'<line x1="{px}" x2="{px + pw}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" stroke="{t["grid"]}"/>')
        body.append(_text(px - 10, sy(yv) + 4, f"{yv:g}", 12, t["muted"], anchor="end"))
    body.append(f'<line x1="{px}" x2="{px + pw}" y1="{py + ph}" y2="{py + ph}" stroke="{t["axis"]}"/>')
    for xv in range(0, 11, 2):
        body.append(_text(sx(xv), py + ph + 20, f"{xv}", 12, t["muted"], anchor="middle"))
    body.append(_text(px + pw / 2, py + ph + 42, "input x", 12, t["muted"], anchor="middle"))
    upper = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(f["xs"], f["hi"]))
    lower = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(f["xs"][::-1], f["lo"][::-1]))
    body.append(f'<polygon points="{upper} {lower}" fill="{t["accent"]}" fill-opacity="0.16"/>')
    truthPts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(f["xs"], f["truth"]))
    body.append(f'<polyline points="{truthPts}" fill="none" stroke="{t["muted"]}" stroke-width="1.5" '
                f'stroke-dasharray="5 4"/>')
    meanPts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(f["xs"], f["mean"]))
    body.append(f'<polyline points="{meanPts}" fill="none" stroke="{t["accent"]}" stroke-width="2" '
                f'stroke-linejoin="round"/>')
    for a, b in zip(f["x"], f["y"]):
        body.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4.5" fill="{t["accent2"]}" '
                    f'stroke="{t["surface"]}" stroke-width="2"/>')
    # legend
    lx, ly = px, py - 18
    body.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="4.5" fill="{t["accent2"]}"/>')
    body.append(_text(lx + 16, ly, "14 noisy samples", 12.5, t["ink2"]))
    lx += 136
    body.append(f'<line x1="{lx}" x2="{lx + 20}" y1="{ly - 4}" y2="{ly - 4}" stroke="{t["accent"]}" stroke-width="2"/>')
    body.append(_text(lx + 26, ly, "Kriging mean", 12.5, t["ink2"]))
    lx += 118
    body.append(f'<rect x="{lx}" y="{ly - 10}" width="20" height="12" rx="2" fill="{t["accent"]}" fill-opacity="0.16"/>')
    body.append(_text(lx + 26, ly, "95 % prediction interval", 12.5, t["ink2"]))
    lx += 176
    body.append(f'<line x1="{lx}" x2="{lx + 20}" y1="{ly - 4}" y2="{ly - 4}" stroke="{t["muted"]}" stroke-width="1.5" '
                f'stroke-dasharray="5 4"/>')
    body.append(_text(lx + 26, ly, "truth", 12.5, t["ink2"]))

    # right column: code and capabilities
    rx, ry = 640, 112
    body.append(f'<rect x="{rx}" y="{ry}" width="328" height="74" rx="8" fill="{t["codeBg"]}"/>')
    body.append(_text(rx + 14, ry + 28, 'm = createModel("kriging").fit(x, y)', 12.5, t["ink"], family=MONO))
    body.append(_text(rx + 14, ry + 52, "band = m.predictInterval(xs, 0.95)", 12.5, t["ink"], family=MONO))
    items = [
        ("30+ model types", "polynomial · Kriging · GAM · GLM · RBF · splines"),
        ("", "quantile · mixed · forests · boosting · networks"),
        ("Uncertainty and gradients", "predictInterval · predictGradient"),
        ("Validation", f"exact LOO / k-fold  (this fit: LOO RMSE {f['looRmse']:.3f})"),
        ("Physics constraints", "integrals, bounds, monotone, Buckingham pi"),
        ("Portable", "JSON save / load, no scipy or sklearn"),
    ]
    yy = ry + 112
    for head, detail in items:
        if head:
            yy += 8
            body.append(_text(rx, yy, head, 14, t["ink"], "600"))
            yy += 20
        body.append(_text(rx, yy, detail, 12.5, t["ink2"]))
        yy += 20
    return _svg(474, t, body, "regressionHandler overview")


# ---------------------------------------------------------------------------
# fieldMapping: conservative flux transfer between non-matching meshes
# ---------------------------------------------------------------------------
def _fieldMappingFacts():
    from pythonLibs.fieldMapping import SurfaceFluxMapper, planeSurface

    src = planeSurface(n=(23, 17), cellType="triangle", jitter=0.3, seed=0)
    tgt = planeSurface(n=(7, 9), cellType="quad", jitter=0.2, seed=1)
    flux = lambda p: np.sin(6.0 * p[:, 0]) + 0.3 * p[:, 1] - 0.15             # noqa: E731, changes sign
    qSrc = flux(src.cellCentroids())
    res = SurfaceFluxMapper(src, tgt).map(qSrc, "cell")
    d = res.diagnostics
    # naive comparison: sample the source flux of the nearest source cell at each target centroid
    cs, ct = src.cellCentroids()[:, :2], tgt.cellCentroids()[:, :2]
    nearest = np.argmin(((ct[:, None, :] - cs[None, :, :]) ** 2).sum(-1), axis=1)
    heatNaive = qSrc[nearest] * tgt.cellMeasures()
    naiveIn = heatNaive[heatNaive > 0].sum()
    naiveOut = heatNaive[heatNaive < 0].sum()
    return dict(src=src, tgt=tgt, qSrc=qSrc, qTgt=res.cellFlux, d=d,
                naiveErrIn=naiveIn / d["sourceHeatIn"] - 1.0, naiveErrOut=naiveOut / d["sourceHeatOut"] - 1.0,
                mappedErrIn=d["relativeErrorIn"], mappedErrOut=d["relativeErrorOut"])


def _mix(c1, c2, w):
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)], float)
    r = np.round(a + (b - a) * w).astype(int)
    return "#" + "".join(f"{v:02x}" for v in r)


def _diverging(t, v, vmax):
    w = float(np.clip(v / vmax, -1.0, 1.0))
    return _mix(t["divMid"], t["divPos"], w) if w >= 0 else _mix(t["divMid"], t["divNeg"], -w)


def _meshPanel(t, mesh, values, vmax, x0, y0, size):
    out = []
    pts = mesh.points[:, :2]
    for c in range(mesh.nCells):
        ids = mesh.connectivity[mesh.offsets[c]:mesh.offsets[c + 1]]
        poly = " ".join(f"{x0 + p[0] * size:.1f},{y0 + (1 - p[1]) * size:.1f}" for p in pts[ids])
        out.append(f'<polygon points="{poly}" fill="{_diverging(t, values[c], vmax)}" stroke="{t["surface"]}" '
                   f'stroke-width="0.8" stroke-linejoin="round"/>')
    return out


def fieldMappingImage(t, f):
    body = _header(t, "fieldMapping — conservative field transfer between non-matching meshes",
                   "Heat flux from a fine CFD surface onto a coarse structural mesh: incoming and outgoing heat both "
                   "kept exactly.")
    size, y0 = 272, 138
    vmax = float(np.max(np.abs(f["qSrc"])))
    ax, bx = 32, 392
    body.append(_text(ax, y0 - 14, f"Source: {f['src'].nCells} triangles (e.g. CFD)", 13, t["ink2"], "600"))
    body.extend(_meshPanel(t, f["src"], f["qSrc"], vmax, ax, y0, size))
    body.append(_text(bx, y0 - 14, f"Target: {f['tgt'].nCells} distorted quads (e.g. FEM)", 13, t["ink2"], "600"))
    body.extend(_meshPanel(t, f["tgt"], f["qTgt"], vmax, bx, y0, size))
    mx, my = ax + size + 44, y0 + size / 2
    body.append(f'<path d="M{mx - 30},{my} L{mx + 26},{my}" stroke="{t["accent"]}" stroke-width="2"/>')
    body.append(f'<path d="M{mx + 18},{my - 6} L{mx + 27},{my} L{mx + 18},{my + 6}" fill="none" '
                f'stroke="{t["accent"]}" stroke-width="2" stroke-linejoin="round"/>')
    body.append(_text(mx, my - 14, "map", 12, t["accent"], "600", anchor="middle"))
    # colour legend
    gy = y0 + size + 26
    steps = 40
    for i in range(steps):
        v = -vmax + 2 * vmax * (i + 0.5) / steps
        body.append(f'<rect x="{ax + i * 8}" y="{gy}" width="8.4" height="10" fill="{_diverging(t, v, vmax)}"/>')
    body.append(_text(ax, gy + 28, "heat out", 12, t["muted"]))
    body.append(_text(ax + steps * 4, gy + 28, "0", 12, t["muted"], anchor="middle"))
    body.append(_text(ax + steps * 8, gy + 28, "heat in", 12, t["muted"], anchor="end"))
    body.append(_text(bx, gy + 9, "surface heat flux q", 12, t["muted"]))

    # conservation table
    d = f["d"]
    rx = 716
    body.append(_text(rx, y0 - 14, "Total heat on the target", 13, t["ink2"], "600"))
    head = y0 + 14
    body.append(_text(rx + 104, head, "in", 12, t["muted"], anchor="end"))
    body.append(_text(rx + 180, head, "out", 12, t["muted"], anchor="end"))
    body.append(f'<line x1="{rx}" x2="968" y1="{head + 8}" y2="{head + 8}" stroke="{t["grid"]}"/>')

    def pct(v):
        return "< 1e-12" if abs(v) < 1e-12 else f"{100 * v:+.1f} %"

    rows = [("source", f"{d['sourceHeatIn']:.4f}", f"{d['sourceHeatOut']:.4f}", t["ink"]),
            ("mapped", f"{d['targetHeatIn']:.4f}", f"{d['targetHeatOut']:.4f}", t["ink"])]
    yy = head + 30
    for name, a, b, col in rows:
        body.append(_text(rx, yy, name, 13, t["ink2"]))
        body.append(_text(rx + 104, yy, a, 13, col, family=MONO, anchor="end"))
        body.append(_text(rx + 180, yy, b, 13, col, family=MONO, anchor="end"))
        yy += 24
    yy += 12
    body.append(_text(rx, yy, "Error of the heat totals", 13, t["ink2"], "600"))
    yy += 24
    body.append(_text(rx, yy, "SurfaceFluxMapper", 13, t["ink"]))
    yy += 20
    body.append(_text(968, yy, f"in {pct(f['mappedErrIn'])} · out {pct(f['mappedErrOut'])}", 12.5, t["good"],
                      family=MONO, anchor="end"))
    yy += 22
    body.append(_text(rx, yy, "nearest-cell sampling", 13, t["ink"]))
    yy += 20
    body.append(_text(968, yy, f"in {pct(f['naiveErrIn'])} · out {pct(f['naiveErrOut'])}", 12.5, t["ink2"],
                      family=MONO, anchor="end"))
    yy += 34
    for line in ("No heat of the wrong sign on any facet.", "VTU / PVD in and out, any VTK cell,",
                 "polyhedra; 1-D layer profiles to 3-D too."):
        body.append(_text(rx, yy, line, 12.5, t["muted"]))
        yy += 19
    return _svg(488, t, body, "fieldMapping overview")


def main():
    facts = {"toolBase": _toolBaseFacts(), "regressionHandler": _regressionFacts(),
             "fieldMapping": _fieldMappingFacts()}
    makers = {"toolBase": toolBaseImage, "regressionHandler": regressionImage, "fieldMapping": fieldMappingImage}
    for name, make in makers.items():
        for mode, theme in THEMES.items():
            print(_write(name, mode, make(theme, facts[name])))


if __name__ == "__main__":
    main()
