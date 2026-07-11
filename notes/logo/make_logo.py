"""Generate vector FRIDOM logo drafts from a computed flow field.

The construction mirrors the original raster logo: the word FRIDOM is
turned into a smooth streamfunction (a Gaussian-blurred glyph mask),
and everything visible in the SVGs is derived from that field —
streamlines are psi-contours, arrows follow the velocity
(u, v) = (dpsi/dy, -dpsi/dx).  No shape is drawn by hand; tweak the
knobs below and re-run.

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
TEXT = " ".join("FRIDOM")  # thin-space tracking between letters
FONT = {"family": "DejaVu Sans", "weight": "bold", "style": "oblique"}
FONT_SIZE = 100.0   # glyph size in logo units
PAD = 36.0          # canvas margin around the glyphs
DX = 0.4            # grid spacing (logo units)
SIGMA = 2.6         # Gaussian blur width -> jet thickness
WARP_AMP = 1.2      # fluid-like domain warp amplitude (0 = off)
WARP_LAM = 70.0     # domain warp wavelength
NODE_STEP = 3.0     # arc-length between SVG path nodes (units)

# draft A: nested streamlines (level, color, stroke width)
LEVELS_A = [
    (0.08, "#f0d5a8", 2.0),
    (0.35, "#e39a5b", 2.5),
    (0.58, "#c93a2b", 4.0),
    (0.82, "#7e1a15", 2.0),
]
ARROW_LEVEL_A = 0.35    # streamline that carries the arrowheads
ARROW_SPACING_A = 48.0  # arc length between arrowheads

# draft B: vector remake of the original quiver look
ENV_LEVEL_B = 0.06      # cream envelope
RED_LEVEL_B = 0.45      # red letter band
CORE_LEVEL_B = 0.80     # pale letter core
GRID_STEP_B = 12.0      # quiver grid spacing
ARROW_MIN_SPEED = 0.22  # fraction of max speed below which a dot is drawn


# ================================================================
#  Field construction
# ================================================================
def domain_warp(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,
                                                        np.ndarray]:
    """Gentle deterministic warp so contours look advected, not typeset."""
    a, lam = WARP_AMP, WARP_LAM
    wx = a * (np.sin(2 * np.pi * y / lam + 1.7)
              + 0.6 * np.sin(2 * np.pi * y / (0.43 * lam) + 4.1))
    wy = a * (np.sin(2 * np.pi * x / lam + 0.6)
              + 0.6 * np.sin(2 * np.pi * x / (0.37 * lam) + 2.9))
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


def build_field() -> dict:
    """Rasterize the glyph mask, blur it into psi, compute velocity."""
    prop = FontProperties(**FONT)
    tp = TextPath((0, 0), TEXT, size=FONT_SIZE, prop=prop)
    ext = tp.get_extents()

    x0, y0 = ext.x0 - PAD, ext.y0 - PAD
    width, height = ext.width + 2 * PAD, ext.height + 2 * PAD
    x1 = x0 + np.arange(int(round(width / DX))) * DX
    y1 = y0 + np.arange(int(round(height / DX))) * DX
    mask = rasterize_mask(tp, x0, y0, width, height, len(x1), len(y1))

    # gaussian blur via fft
    ky = np.fft.fftfreq(len(y1), d=DX)
    kx = np.fft.rfftfreq(len(x1), d=DX)
    kernel = np.exp(-2 * np.pi**2 * SIGMA**2
                    * (kx[None, :]**2 + ky[:, None]**2))
    psi = np.fft.irfft2(np.fft.rfft2(mask) * kernel, s=mask.shape)
    psi /= psi.max()

    f = {"x": x1, "y": y1, "x0": x0, "y0": y0,
         "width": width, "height": height}
    if WARP_AMP > 0:
        xg, yg = np.meshgrid(x1, y1)
        xw, yw = domain_warp(xg, yg)
        pts = np.column_stack([xw.ravel(), yw.ravel()])
        psi = interp(psi, f, pts).reshape(psi.shape)

    dpsi_dy, dpsi_dx = np.gradient(psi, DX, DX)
    f.update(psi=psi, u=dpsi_dy, v=-dpsi_dx)
    return f


def interp(field2d: np.ndarray, f: dict, pts: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a gridded field at (x, y) points."""
    fx = np.clip((pts[:, 0] - f["x0"]) / DX, 0, len(f["x"]) - 1.001)
    fy = np.clip((pts[:, 1] - f["y0"]) / DX, 0, len(f["y"]) - 1.001)
    i, j = fy.astype(int), fx.astype(int)
    ty, tx = fy - i, fx - j
    z = field2d
    return ((1 - ty) * (1 - tx) * z[i, j] + (1 - ty) * tx * z[i, j + 1]
            + ty * (1 - tx) * z[i + 1, j] + ty * tx * z[i + 1, j + 1])


# ================================================================
#  Geometry -> SVG paths
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
    """Data coords (y up) -> SVG coords (y down, origin top-left)."""
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


# ================================================================
#  SVG documents
# ================================================================
def svg_header(f: dict) -> str:
    w, h = f["width"], f["height"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w:.0f}" height="{h:.0f}" '
            f'viewBox="0 0 {w:.2f} {h:.2f}">\n')


def draft_a(f: dict, path: Path) -> None:
    """Nested streamlines wordmark with arrowheads (transparent bg)."""
    parts = [svg_header(f)]
    for level, color, width in LEVELS_A:
        parts.append(f'  <g id="psi-{level}" fill="none" stroke="{color}" '
                     f'stroke-width="{width}" stroke-linejoin="round">\n')
        parts.append(f'    <path d="{level_path(f, level)}"/>\n')
        parts.append("  </g>\n")
    parts.append('  <g id="arrows" fill="#3a3f45">\n')
    for x, y, ang in arrow_transforms(f, ARROW_LEVEL_A, ARROW_SPACING_A):
        parts.append(f'    <path transform="translate({x:.2f} {y:.2f}) '
                     f'rotate({ang:.1f})" '
                     f'd="M 4.6 0 L -3.4 2.9 L -3.4 -2.9 Z"/>\n')
    parts.append("  </g>\n</svg>\n")
    path.write_text("".join(parts))


def draft_b(f: dict, path: Path) -> None:
    """Vector remake of the original: cream envelope, red band, quiver."""
    parts = [svg_header(f)]
    parts.append('  <defs>\n'
                 '    <path id="tri" d="M 2.5 0 L -2.1 1.6 L -2.1 -1.6 Z"'
                 ' fill="#363b41"/>\n'
                 '    <circle id="dot" r="0.9" fill="#c9c2b4"/>\n'
                 '  </defs>\n')
    parts.append(f'  <path id="envelope" fill="#faf1dc" fill-rule="evenodd"'
                 f' d="{level_path(f, ENV_LEVEL_B)}"/>\n')
    parts.append(f'  <path id="letters" fill="#d0402e" fill-rule="evenodd"'
                 f' stroke="#a62a1e" stroke-width="1.2"'
                 f' d="{level_path(f, RED_LEVEL_B)}"/>\n')
    parts.append(f'  <path id="cores" fill="#f8eeda" fill-rule="evenodd"'
                 f' d="{level_path(f, CORE_LEVEL_B)}"/>\n')

    speed = np.hypot(f["u"], f["v"])
    smax = speed.max()
    parts.append('  <g id="quiver">\n')
    xs = np.arange(f["x0"] + PAD * 0.4, f["x0"] + f["width"] - PAD * 0.4,
                   GRID_STEP_B)
    ys = np.arange(f["y0"] + PAD * 0.4, f["y0"] + f["height"] - PAD * 0.4,
                   GRID_STEP_B)
    for yv in ys:
        pts = np.column_stack([xs, np.full_like(xs, yv)])
        sp = interp(speed, f, pts)
        vx = interp(f["u"], f, pts)
        vy = interp(f["v"], f, pts)
        svg_pts = to_svg(pts, f)
        for k, (x, y) in enumerate(svg_pts):
            if sp[k] > ARROW_MIN_SPEED * smax:
                ang = np.degrees(np.arctan2(-vy[k], vx[k]))
                parts.append(f'    <use href="#tri" transform='
                             f'"translate({x:.2f} {y:.2f}) '
                             f'rotate({ang:.1f})"/>\n')
            else:
                parts.append(f'    <use href="#dot" '
                             f'x="{x:.2f}" y="{y:.2f}"/>\n')
    parts.append("  </g>\n</svg>\n")
    path.write_text("".join(parts))


def main() -> None:
    f = build_field()
    draft_a(f, HERE / "fridom-logo-streamlines.svg")
    draft_b(f, HERE / "fridom-logo-quiver.svg")
    print(f"canvas {f['width']:.0f} x {f['height']:.0f} units")


if __name__ == "__main__":
    main()
