"""Generate vector FRIDOM logo drafts from a computed flow field.

The construction mirrors the original raster logo: each letter is
turned into a smooth streamfunction (a Gaussian-blurred glyph mask),
and everything visible in the SVGs is derived from that field —
streamlines are psi-contours, arrows follow the velocity
(u, v) = (dpsi/dy, -dpsi/dx), and the letter fill is a coarse
model-grid mosaic colored by a quantized, wider-blurred psi.
No shape is drawn by hand; tweak the knobs below and re-run.

Each letter is a standalone ``<g id="letter-X" transform="translate(...)">``
group, so letters can be moved individually in Inkscape.  Letter offsets
snap to the cell grid so the mosaics of all letters stay aligned.

The shipped logo, ``fridom-final.svg``, is a hand-tuned (Inkscape) pick
from the ``fridom-logo-v3-*`` variant family this script writes;
regenerated variants are gitignored.

Usage::

    python make_logo.py            # writes SVGs next to this script
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from contourpy import LineType, contour_generator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath

HERE = Path(__file__).parent

# ================================================================
#  Knobs
# ================================================================
WORD = "FRIDOM"
FONT = {"family": "DejaVu Sans", "weight": "bold", "style": "oblique"}
FONT_SIZE = 100.0   # glyph size in logo units
PAD_L = 26.0        # horizontal margin around each letter
PAD_V = 30.0        # vertical canvas margin
DX = 0.4            # grid spacing (logo units)
SIGMA = 2.6         # Gaussian blur width -> jet thickness
WARP_AMP = 1.3      # fluid-like domain warp amplitude (0 = off)
WARP_LAM = 75.0     # domain warp wavelength (long: no small-scale twists)
NODE_STEP = 9.0     # arc-length between SVG path nodes (units)

# ink-to-ink letter spacing of the generated variants
SPACINGS = {"tight": 4.0, "mid": 10.0, "wide": 18.0}

# streamlines: (psi level, color, stroke width), outside -> inside.
# all colors are mid-luminance cyans so one svg works on light and
# dark backgrounds alike.
LEVELS = [
    (0.06, "#43c6d8", 2.0),
    (0.58, "#12869c", 3.0),
]
ARROW_LEVEL = 0.06      # streamline that carries the arrowheads
ARROW_SPACING = 55.0    # arc length between arrowheads
ARROW_COLOR = "#43c6d8"

# discretized letter fill: coarse-grid cells inside the letters
# (psi >= CELL_MIN), colored by a wider-blurred psi so the shading
# varies across the letter body instead of saturating.
CELL_STEP = 6.5         # model-grid cell size
CELL_GAP = 0.9          # gap between cells (background shows through)
CELL_MIN = 0.50         # psi threshold for a cell to belong to a letter
SIGMA_FILL = 6.0        # blur width of the fill-shading field
CELL_BINS = [           # (min psi_fill, color), brighter towards the core
    (0.00, "#157887"),
    (0.55, "#1ba3b4"),
    (0.75, "#3ecfdf"),
]

# faint "inactive" cells outside the letters (the *-cells variants);
# neutral mid-gray at low opacity reads on light and dark backgrounds
BG_CELL_COLOR = "#888888"
BG_CELL_OPACITY = 0.09


# ================================================================
#  Field construction (one field per letter)
# ================================================================
def domain_warp(x: np.ndarray, y: np.ndarray,
                phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Gentle deterministic warp so contours look advected, not typeset."""
    a, lam = WARP_AMP, WARP_LAM
    wx = a * np.sin(2 * np.pi * y / lam + 1.7 + phase)
    wy = a * np.sin(2 * np.pi * x / lam + 0.6 + 2.3 * phase)
    return x + wx, y + wy


def rasterize_mask(tp: TextPath, x0: float, y0: float, width: float,
                   height: float, nx: int, ny: int) -> np.ndarray:
    """Rasterize the glyph path with Agg (handles letter counters)."""
    dpi = 100
    fig = Figure(figsize=(nx / dpi, ny / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(x0, x0 + width)
    ax.set_ylim(y0, y0 + height)
    ax.set_axis_off()
    ax.add_patch(PathPatch(tp, facecolor="black", edgecolor="none"))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    # flip so that row 0 corresponds to y0 (grid convention, y up)
    return (rgba[::-1, :, 0] < 128).astype(float)


def interp(field2d: np.ndarray, f: dict, pts: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a gridded field at (x, y) points."""
    fx = np.clip((pts[:, 0] - f["x0"]) / DX, 0, len(f["x"]) - 1.001)
    fy = np.clip((pts[:, 1] - f["y0"]) / DX, 0, len(f["y"]) - 1.001)
    i, j = fy.astype(int), fx.astype(int)
    ty, tx = fy - i, fx - j
    z = field2d
    return ((1 - ty) * (1 - tx) * z[i, j] + (1 - ty) * tx * z[i, j + 1]
            + ty * (1 - tx) * z[i + 1, j] + ty * tx * z[i + 1, j + 1])


def build_letter(tp: TextPath, y0: float, height: float,
                 phase: float) -> dict:
    """Streamfunction, fill field and velocity for a single letter."""
    ext = tp.get_extents()
    x0 = ext.x0 - PAD_L
    width = ext.width + 2 * PAD_L
    x1 = x0 + np.arange(int(round(width / DX))) * DX
    y1 = y0 + np.arange(int(round(height / DX))) * DX
    mask = rasterize_mask(tp, x0, y0, width, height, len(x1), len(y1))

    # gaussian blurs via fft
    ky = np.fft.fftfreq(len(y1), d=DX)
    kx = np.fft.rfftfreq(len(x1), d=DX)
    k2 = kx[None, :]**2 + ky[:, None]**2
    mask_hat = np.fft.rfft2(mask)

    def blur(sigma: float) -> np.ndarray:
        kernel = np.exp(-2 * np.pi**2 * sigma**2 * k2)
        field = np.fft.irfft2(mask_hat * kernel, s=mask.shape)
        return field / field.max()

    psi = blur(SIGMA)
    psi_fill = blur(SIGMA_FILL)

    f = {"x": x1, "y": y1, "x0": x0, "y0": y0,
         "width": width, "height": height,
         "ink_x0": ext.x0, "ink_x1": ext.x1}
    if WARP_AMP > 0:
        xg, yg = np.meshgrid(x1, y1)
        xw, yw = domain_warp(xg, yg, phase)
        pts = np.column_stack([xw.ravel(), yw.ravel()])
        psi = interp(psi, f, pts).reshape(psi.shape)
        psi_fill = interp(psi_fill, f, pts).reshape(psi_fill.shape)

    dpsi_dy, dpsi_dx = np.gradient(psi, DX, DX)
    f.update(psi=psi, psi_fill=psi_fill, u=dpsi_dy, v=-dpsi_dx)
    return f


def build_letters() -> list[dict]:
    """One field per letter of WORD, sharing a common vertical frame."""
    prop = FontProperties(**FONT)
    tps = [TextPath((0, 0), ch, size=FONT_SIZE, prop=prop) for ch in WORD]
    y0 = min(tp.get_extents().y0 for tp in tps) - PAD_V
    y1 = max(tp.get_extents().y1 for tp in tps) + PAD_V
    letters = []
    for idx, (ch, tp) in enumerate(zip(WORD, tps)):
        f = build_letter(tp, y0, y1 - y0, phase=1.3 * idx)
        f["char"] = ch
        letters.append(f)
    return letters


# ================================================================
#  Geometry -> SVG paths (local letter coordinates)
# ================================================================
def contour_loops(f: dict, level: float) -> list[np.ndarray]:
    """Closed psi-contour loops at a level, in data coordinates."""
    cg = contour_generator(x=f["x"], y=f["y"], z=f["psi"],
                           line_type=LineType.Separate)
    return [ln for ln in cg.lines(level) if len(ln) > 8]


def resample(pts: np.ndarray, step: float) -> np.ndarray:
    """Resample a closed polyline at uniform arc length."""
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(8, int(round(total / step)))
    si = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack([np.interp(si, s, pts[:, 0]),
                            np.interp(si, s, pts[:, 1])])


def to_svg(pts: np.ndarray, f: dict) -> np.ndarray:
    """Data coords (y up) -> local SVG coords (y down, origin top-left)."""
    out = pts.copy()
    out[:, 0] -= f["x0"]
    out[:, 1] = f["height"] - (pts[:, 1] - f["y0"])
    return out


def smooth_path(pts: np.ndarray) -> str:
    """Closed Catmull-Rom spline through the points, as SVG cubics."""
    n = len(pts)
    p = lambda i: pts[i % n]  # noqa: E731
    d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    for i in range(n):
        p0, p1, p2, p3 = p(i - 1), p(i), p(i + 1), p(i + 2)
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d.append(f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f}"
                 f" {p2[0]:.2f} {p2[1]:.2f}")
    return " ".join(d) + " Z"


def level_path(f: dict, level: float) -> str:
    """All loops of a level as one path (evenodd handles the holes)."""
    return " ".join(smooth_path(to_svg(resample(ln, NODE_STEP), f))
                    for ln in contour_loops(f, level))


def arrow_transforms(f: dict, level: float,
                     spacing: float) -> list[tuple[float, float, float]]:
    """(x, y, angle_deg) for arrowheads riding a streamline."""
    out = []
    for ln in contour_loops(f, level):
        loop = resample(ln, 2.0)
        seg = np.hypot(*np.diff(np.vstack([loop, loop[:1]]), axis=0).T)
        total = seg.sum()
        if total < 1.5 * spacing:
            continue
        s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
        n_arrows = int(total // spacing)
        for k in range(n_arrows):
            idx = np.searchsorted(s, 0.5 * spacing + k * total / n_arrows)
            pt = loop[min(idx, len(loop) - 1)][None, :]
            vx = interp(f["u"], f, pt)[0]
            vy = interp(f["v"], f, pt)[0]
            sp = to_svg(pt, f)[0]
            ang = np.degrees(np.arctan2(-vy, vx))  # svg y is flipped
            out.append((sp[0], sp[1], ang))
    return out


def aligned_centers(lo: float, hi: float) -> np.ndarray:
    """Cell centers k*CELL_STEP + CELL_STEP/2 covering [lo, hi]."""
    k0 = int(np.ceil((lo - CELL_STEP / 2) / CELL_STEP))
    k1 = int(np.floor((hi - CELL_STEP / 2) / CELL_STEP))
    return (np.arange(k0, k1 + 1) + 0.5) * CELL_STEP


def cell_rects(f: dict) -> list[tuple[float, float, str]]:
    """(x, y, color) of discretized-fill cells covering one letter."""
    size = CELL_STEP - CELL_GAP
    xs = aligned_centers(f["x0"], f["x0"] + f["width"])
    ys = aligned_centers(f["y0"], f["y0"] + f["height"])
    out = []
    for yv in ys:
        pts = np.column_stack([xs, np.full_like(xs, yv)])
        ps = interp(f["psi"], f, pts)
        pf = interp(f["psi_fill"], f, pts)
        svg_pts = to_svg(pts, f)
        for k, (cx, cy) in enumerate(svg_pts):
            if ps[k] < CELL_MIN:
                continue
            color = CELL_BINS[0][1]
            for level, col in CELL_BINS:
                if pf[k] >= level:
                    color = col
            out.append((cx - size / 2, cy - size / 2, color))
    return out


# ================================================================
#  SVG documents
# ================================================================
def letter_group(f: dict, tx: float) -> str:
    """One letter as a movable group in local coordinates."""
    size = CELL_STEP - CELL_GAP
    parts = [f'  <g id="letter-{f["char"]}" '
             f'transform="translate({tx:.2f} 0)">\n']
    parts.append('    <g class="grid-fill">\n')
    for x, y, color in cell_rects(f):
        parts.append(f'      <rect x="{x:.2f}" y="{y:.2f}" '
                     f'width="{size}" height="{size}" fill="{color}"/>\n')
    parts.append("    </g>\n")
    for level, color, width in LEVELS:
        parts.append(f'    <g class="psi-{level}" fill="none" '
                     f'stroke="{color}" stroke-width="{width}" '
                     f'stroke-linejoin="round">\n')
        parts.append(f'      <path d="{level_path(f, level)}"/>\n')
        parts.append("    </g>\n")
    parts.append(f'    <g class="arrows" fill="{ARROW_COLOR}">\n')
    for x, y, ang in arrow_transforms(f, ARROW_LEVEL, ARROW_SPACING):
        parts.append(f'      <path transform="translate({x:.2f} {y:.2f}) '
                     f'rotate({ang:.1f})" '
                     f'd="M 3.4 0 L -2.5 2.1 L -2.5 -2.1 Z"/>\n')
    parts.append("    </g>\n  </g>\n")
    return "".join(parts)


def wordmark(letters: list[dict], gap: float, bg_cells: bool,
             path: Path) -> None:
    """Assemble one spacing variant; letter offsets snap to the grid."""
    height = letters[0]["height"]
    offsets = []
    cursor = PAD_L
    for f in letters:
        tx = cursor - f["ink_x0"] + f["x0"]  # local origin -> global
        # snap so the mosaics of all letters share one global grid
        tx = f["x0"] + round((tx - f["x0"]) / CELL_STEP) * CELL_STEP
        offsets.append(tx)
        cursor = tx + (f["ink_x1"] - f["x0"]) + gap
    width = offsets[-1] + letters[-1]["width"]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{width:.0f}" height="{height:.0f}" '
             f'viewBox="0 0 {width:.2f} {height:.2f}">\n']
    if bg_cells:
        size = CELL_STEP - CELL_GAP
        active = set()
        for f, tx in zip(letters, offsets):
            for x, y, _ in cell_rects(f):
                active.add((round(x + tx, 1), round(y, 1)))
        parts.append(f'  <g id="grid-bg" fill="{BG_CELL_COLOR}" '
                     f'fill-opacity="{BG_CELL_OPACITY}">\n')
        y0 = letters[0]["y0"]
        ys = height - (aligned_centers(y0, y0 + height) - y0)
        for xc in aligned_centers(0.0, width):
            for yc in ys:
                x, y = xc - size / 2, yc - size / 2
                if (round(x, 1), round(y, 1)) in active:
                    continue
                parts.append(f'    <rect x="{x:.2f}" y="{y:.2f}" '
                             f'width="{size}" height="{size}"/>\n')
        parts.append("  </g>\n")
    for f, tx in zip(letters, offsets):
        parts.append(letter_group(f, tx))
    parts.append("</svg>\n")
    path.write_text("".join(parts))


def main() -> None:
    letters = build_letters()
    for name, gap in SPACINGS.items():
        for bg_cells in (False, True):
            suffix = f"{name}-cells" if bg_cells else name
            out = HERE / f"fridom-logo-v3-{suffix}.svg"
            wordmark(letters, gap, bg_cells, out)
            print(out.name)


if __name__ == "__main__":
    main()
