"""
RUL plotting (PHASE 11 extension).

Renders the four required plots as **SVG** files, no third-party
plotting dependency. SVG is a text format that any browser /
dashboard / doc-renderer can display, and the charts are
deterministic given the same inputs (no RNG, no headless backend
quirks).

The four plots are:

1. **Health Index vs time** — single line of the overall
   :class:`~backend.health.HealthIndex.overall_score` over the
   mission. Empty input renders a "no data" placeholder.
2. **RUL vs time** — line of :class:`RulEstimate.tte_hours_central`
   over the mission, with the 5/95 % interval as a shaded band.
3. **Actual vs predicted RUL** — scatter of ``truth`` vs
   ``central``, with the ``y = x`` reference line.
4. **Prediction uncertainty** — error-bar plot of ``(lower,
   upper)`` intervals around the central estimate, sorted by
   central value so the operator can see the spread.

Each function returns the path it wrote to, for chaining.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Sequence, Tuple

from backend.health import HealthIndex
from .types import RulEstimate


# Default canvas size.
DEFAULT_WIDTH: float = 800.0
DEFAULT_HEIGHT: float = 400.0
MARGIN_LEFT: float = 60.0
MARGIN_RIGHT: float = 20.0
MARGIN_TOP: float = 30.0
MARGIN_BOTTOM: float = 40.0


def _ensure_path(path: Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _svg_header(width: float, height: float) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:g} {height:g}" '
        f'width="{width:g}" height="{height:g}">\n'
        f'<rect x="0" y="0" width="{width:g}" height="{height:g}" '
        f'fill="white"/>\n'
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _empty_svg(width: float, height: float, title: str) -> str:
    out = _svg_header(width, height)
    out += (
        f'<text x="{width/2:g}" y="{height/2:g}" '
        f'text-anchor="middle" dominant-baseline="middle" '
        f'font-family="sans-serif" font-size="14" fill="#666">'
        f'no data</text>\n'
    )
    out += (
        f'<text x="{width/2:g}" y="20" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#333">{escape(title)}</text>\n'
    )
    out += _svg_footer()
    return out


def _line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_vals: Sequence[float],
    y_lines: Sequence[Tuple[str, Sequence[float], str]],  # (name, y, color)
    y_band: Tuple[Sequence[float], Sequence[float], str] = None,  # (lo, hi, color)
    y_min: float = None,
    y_max: float = None,
    width: float = DEFAULT_WIDTH,
    height: float = DEFAULT_HEIGHT,
) -> str:
    """Render a line chart with optional shaded band.

    ``y_lines`` is a list of (name, y_values, color) tuples.
    ``y_band`` is (lower, upper, color) — the band is drawn first,
    then the lines on top.
    """
    out = _svg_header(width, height)
    out += (
        f'<text x="{width/2:g}" y="20" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#333">{escape(title)}</text>\n'
    )
    if not x_vals:
        out += _svg_footer()
        return out
    plot_w = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = height - MARGIN_TOP - MARGIN_BOTTOM
    # Axes (origin at top-left of plot region).
    x_min = min(x_vals)
    x_max = max(x_vals)
    x_span = max(1e-9, x_max - x_min)
    all_y: list = []
    for _, y, _ in y_lines:
        all_y.extend(y)
    if y_band is not None:
        all_y.extend(y_band[0])
        all_y.extend(y_band[1])
    if y_min is None:
        y_min = min(all_y) if all_y else 0.0
    if y_max is None:
        y_max = max(all_y) if all_y else 1.0
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    y_span = max(1e-9, y_max - y_min)
    out += (
        f'<line x1="{MARGIN_LEFT:g}" y1="{MARGIN_TOP:g}" '
        f'x2="{MARGIN_LEFT:g}" y2="{MARGIN_TOP+plot_h:g}" '
        f'stroke="#333" stroke-width="1"/>\n'
        f'<line x1="{MARGIN_LEFT:g}" y1="{MARGIN_TOP+plot_h:g}" '
        f'x2="{MARGIN_LEFT+plot_w:g}" y2="{MARGIN_TOP+plot_h:g}" '
        f'stroke="#333" stroke-width="1"/>\n'
    )
    # Y axis ticks (5 ticks).
    for i in range(6):
        yv = y_min + (y_span * i) / 5.0
        ypx = MARGIN_TOP + plot_h - (yv - y_min) / y_span * plot_h
        out += (
            f'<line x1="{MARGIN_LEFT-4:g}" y1="{ypx:g}" '
            f'x2="{MARGIN_LEFT:g}" y2="{ypx:g}" stroke="#333"/>\n'
            f'<text x="{MARGIN_LEFT-6:g}" y="{ypx+4:g}" '
            f'text-anchor="end" font-family="sans-serif" '
            f'font-size="10" fill="#333">{yv:.2g}</text>\n'
        )
    # X axis ticks (5 ticks).
    for i in range(6):
        xv = x_min + (x_span * i) / 5.0
        xpx = MARGIN_LEFT + (xv - x_min) / x_span * plot_w
        out += (
            f'<line x1="{xpx:g}" y1="{MARGIN_TOP+plot_h:g}" '
            f'x2="{xpx:g}" y2="{MARGIN_TOP+plot_h+4:g}" '
            f'stroke="#333"/>\n'
            f'<text x="{xpx:g}" y="{MARGIN_TOP+plot_h+16:g}" '
            f'text-anchor="middle" font-family="sans-serif" '
            f'font-size="10" fill="#333">{xv:.2g}</text>\n'
        )
    # Shaded band first.
    if y_band is not None:
        lo, hi, color = y_band
        # Build a polygon.
        pts_up = []
        pts_dn = []
        for i, xv in enumerate(x_vals):
            xpx = MARGIN_LEFT + (xv - x_min) / x_span * plot_w
            ypx_up = MARGIN_TOP + plot_h - (hi[i] - y_min) / y_span * plot_h
            pts_up.append(f"{xpx:g},{ypx_up:g}")
            ypx_dn = MARGIN_TOP + plot_h - (lo[i] - y_min) / y_span * plot_h
            pts_dn.append(f"{xpx:g},{ypx_dn:g}")
        points = " ".join(pts_up) + " " + " ".join(reversed(pts_dn))
        out += (
            f'<polygon points="{points}" fill="{color}" '
            f'fill-opacity="0.2" stroke="none"/>\n'
        )
    # Then each line.
    for name, y, color in y_lines:
        pts = []
        for i, xv in enumerate(x_vals):
            xpx = MARGIN_LEFT + (xv - x_min) / x_span * plot_w
            ypx = MARGIN_TOP + plot_h - (y[i] - y_min) / y_span * plot_h
            pts.append(f"{xpx:g},{ypx:g}")
        out += (
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{color}" stroke-width="1.5"/>\n'
        )
    # Axis labels.
    out += (
        f'<text x="{MARGIN_LEFT+plot_w/2:g}" y="{height-6:g}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#333">{escape(x_label)}</text>\n'
    )
    out += (
        f'<text x="14" y="{MARGIN_TOP+plot_h/2:g}" '
        f'transform="rotate(-90 14 {MARGIN_TOP+plot_h/2:g})" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#333">{escape(y_label)}</text>\n'
    )
    # Legend (if multiple lines).
    if len(y_lines) > 1 or y_band is not None:
        lx = MARGIN_LEFT + 8
        ly = MARGIN_TOP + 8
        offset = 0
        for name, _, color in y_lines:
            out += (
                f'<line x1="{lx:g}" y1="{ly+offset:g}" '
                f'x2="{lx+18:g}" y2="{ly+offset:g}" '
                f'stroke="{color}" stroke-width="1.5"/>\n'
                f'<text x="{lx+22:g}" y="{ly+offset+4:g}" '
                f'font-family="sans-serif" font-size="10" fill="#333">'
                f'{escape(name)}</text>\n'
            )
            offset += 14
        if y_band is not None:
            color = y_band[2]
            out += (
                f'<rect x="{lx:g}" y="{ly+offset-5:g}" width="18" '
                f'height="10" fill="{color}" fill-opacity="0.2" '
                f'stroke="none"/>\n'
                f'<text x="{lx+22:g}" y="{ly+offset+4:g}" '
                f'font-family="sans-serif" font-size="10" fill="#333">'
                f'5/95 interval</text>\n'
            )
    out += _svg_footer()
    return out


def _scatter_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    ref_line: bool = True,
    width: float = 550.0,
    height: float = 550.0,
) -> str:
    out = _svg_header(width, height)
    out += (
        f'<text x="{width/2:g}" y="20" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#333">{escape(title)}</text>\n'
    )
    if not x_vals:
        out += _svg_footer()
        return out
    plot_w = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = height - MARGIN_TOP - MARGIN_BOTTOM
    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    pad = max(1.0, 0.05 * (hi - lo))
    x_lo, x_hi = lo - pad, hi + pad
    y_lo, y_hi = lo - pad, hi + pad
    x_span = max(1e-9, x_hi - x_lo)
    y_span = max(1e-9, y_hi - y_lo)
    out += (
        f'<line x1="{MARGIN_LEFT:g}" y1="{MARGIN_TOP:g}" '
        f'x2="{MARGIN_LEFT:g}" y2="{MARGIN_TOP+plot_h:g}" '
        f'stroke="#333" stroke-width="1"/>\n'
        f'<line x1="{MARGIN_LEFT:g}" y1="{MARGIN_TOP+plot_h:g}" '
        f'x2="{MARGIN_LEFT+plot_w:g}" y2="{MARGIN_TOP+plot_h:g}" '
        f'stroke="#333" stroke-width="1"/>\n'
    )
    # Ticks.
    for i in range(6):
        xv = x_lo + (x_span * i) / 5.0
        yv = y_lo + (y_span * i) / 5.0
        xpx = MARGIN_LEFT + (xv - x_lo) / x_span * plot_w
        ypx = MARGIN_TOP + plot_h - (yv - y_lo) / y_span * plot_h
        out += (
            f'<line x1="{MARGIN_LEFT-4:g}" y1="{ypx:g}" '
            f'x2="{MARGIN_LEFT:g}" y2="{ypx:g}" stroke="#333"/>\n'
            f'<text x="{MARGIN_LEFT-6:g}" y="{ypx+4:g}" '
            f'text-anchor="end" font-family="sans-serif" '
            f'font-size="10" fill="#333">{yv:.2g}</text>\n'
            f'<line x1="{xpx:g}" y1="{MARGIN_TOP+plot_h:g}" '
            f'x2="{xpx:g}" y2="{MARGIN_TOP+plot_h+4:g}" '
            f'stroke="#333"/>\n'
            f'<text x="{xpx:g}" y="{MARGIN_TOP+plot_h+16:g}" '
            f'text-anchor="middle" font-family="sans-serif" '
            f'font-size="10" fill="#333">{xv:.2g}</text>\n'
        )
    # y=x reference line.
    if ref_line:
        x1 = MARGIN_LEFT
        y1 = MARGIN_TOP + plot_h - (y_lo - y_lo) / y_span * plot_h
        x2 = MARGIN_LEFT + plot_w
        y2 = MARGIN_TOP + plot_h - (y_hi - y_lo) / y_span * plot_h
        out += (
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
            f'stroke="tab:red" stroke-dasharray="4 3" stroke-width="1"/>\n'
        )
    # Scatter points.
    for xv, yv in zip(x_vals, y_vals):
        xpx = MARGIN_LEFT + (xv - x_lo) / x_span * plot_w
        ypx = MARGIN_TOP + plot_h - (yv - y_lo) / y_span * plot_h
        out += (
            f'<circle cx="{xpx:g}" cy="{ypx:g}" r="3" '
            f'fill="tab:blue" fill-opacity="0.7"/>\n'
        )
    out += (
        f'<text x="{MARGIN_LEFT+plot_w/2:g}" y="{height-6:g}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#333">{escape(x_label)}</text>\n'
        f'<text x="14" y="{MARGIN_TOP+plot_h/2:g}" '
        f'transform="rotate(-90 14 {MARGIN_TOP+plot_h/2:g})" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#333">{escape(y_label)}</text>\n'
    )
    out += _svg_footer()
    return out


def _errorbar_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    central: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    truth: Sequence[float],
    width: float = DEFAULT_WIDTH,
    height: float = DEFAULT_HEIGHT,
) -> str:
    out = _svg_header(width, height)
    out += (
        f'<text x="{width/2:g}" y="20" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#333">{escape(title)}</text>\n'
    )
    if not central:
        out += _svg_footer()
        return out
    plot_w = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = height - MARGIN_TOP - MARGIN_BOTTOM
    all_y = list(lower) + list(upper) + list(central) + list(truth)
    y_min = min(all_y)
    y_max = max(all_y)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    pad = 0.05 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    y_span = max(1e-9, y_max - y_min)
    n = len(central)
    out += (
        f'<line x1="{MARGIN_LEFT:g}" y1="{MARGIN_TOP:g}" '
        f'x2="{MARGIN_LEFT:g}" y2="{MARGIN_TOP+plot_h:g}" '
        f'stroke="#333" stroke-width="1"/>\n'
        f'<line x1="{MARGIN_LEFT:g}" y1="{MARGIN_TOP+plot_h:g}" '
        f'x2="{MARGIN_LEFT+plot_w:g}" y2="{MARGIN_TOP+plot_h:g}" '
        f'stroke="#333" stroke-width="1"/>\n'
    )
    for i in range(6):
        yv = y_min + (y_span * i) / 5.0
        ypx = MARGIN_TOP + plot_h - (yv - y_min) / y_span * plot_h
        out += (
            f'<line x1="{MARGIN_LEFT-4:g}" y1="{ypx:g}" '
            f'x2="{MARGIN_LEFT:g}" y2="{ypx:g}" stroke="#333"/>\n'
            f'<text x="{MARGIN_LEFT-6:g}" y="{ypx+4:g}" '
            f'text-anchor="end" font-family="sans-serif" '
            f'font-size="10" fill="#333">{yv:.2g}</text>\n'
        )
    # Error bars + central points.
    for i, (lo, c, hi) in enumerate(zip(lower, central, upper)):
        xpx = MARGIN_LEFT + plot_w * (i + 0.5) / n
        yc = MARGIN_TOP + plot_h - (c - y_min) / y_span * plot_h
        ylo = MARGIN_TOP + plot_h - (lo - y_min) / y_span * plot_h
        yhi = MARGIN_TOP + plot_h - (hi - y_min) / y_span * plot_h
        out += (
            f'<line x1="{xpx:g}" y1="{ylo:g}" x2="{xpx:g}" y2="{yhi:g}" '
            f'stroke="tab:gray" stroke-width="0.8"/>\n'
            f'<line x1="{xpx-2:g}" y1="{ylo:g}" x2="{xpx+2:g}" y2="{ylo:g}" '
            f'stroke="tab:gray" stroke-width="0.8"/>\n'
            f'<line x1="{xpx-2:g}" y1="{yhi:g}" x2="{xpx+2:g}" y2="{yhi:g}" '
            f'stroke="tab:gray" stroke-width="0.8"/>\n'
            f'<circle cx="{xpx:g}" cy="{yc:g}" r="2" fill="tab:purple"/>\n'
        )
    # Truth as red x.
    for i, t in enumerate(truth):
        xpx = MARGIN_LEFT + plot_w * (i + 0.5) / n
        yt = MARGIN_TOP + plot_h - (t - y_min) / y_span * plot_h
        out += (
            f'<line x1="{xpx-3:g}" y1="{yt-3:g}" x2="{xpx+3:g}" y2="{yt+3:g}" '
            f'stroke="tab:red" stroke-width="1.2"/>\n'
            f'<line x1="{xpx-3:g}" y1="{yt+3:g}" x2="{xpx+3:g}" y2="{yt-3:g}" '
            f'stroke="tab:red" stroke-width="1.2"/>\n'
        )
    out += (
        f'<text x="{MARGIN_LEFT+plot_w/2:g}" y="{height-6:g}" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#333">{escape(x_label)}</text>\n'
        f'<text x="14" y="{MARGIN_TOP+plot_h/2:g}" '
        f'transform="rotate(-90 14 {MARGIN_TOP+plot_h/2:g})" '
        f'text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#333">{escape(y_label)}</text>\n'
    )
    out += _svg_footer()
    return out


def plot_health_vs_time(
    health_history: Sequence[HealthIndex],
    path: Path,
    *,
    title: str = "Health Index vs time",
) -> Path:
    """Render the overall health score over time (SVG)."""
    out = _ensure_path(path)
    if not health_history:
        out.write_text(_empty_svg(DEFAULT_WIDTH, DEFAULT_HEIGHT, title),
                       encoding="utf-8")
        return out
    t = [float(h.time_s) for h in health_history]
    s = [float(h.overall_score) for h in health_history]
    svg = _line_chart_svg(
        title=title, x_label="time (s)", y_label="overall health score",
        x_vals=t, y_lines=[("health", s, "tab:blue")],
        y_min=0.0, y_max=1.0,
    )
    out.write_text(svg, encoding="utf-8")
    return out


def plot_rul_vs_time(
    rul_history: Sequence[RulEstimate],
    path: Path,
    *,
    title: str = "RUL vs time",
) -> Path:
    """Render the central TTE estimate (and bounds) over time (SVG)."""
    out = _ensure_path(path)
    if not rul_history:
        out.write_text(_empty_svg(DEFAULT_WIDTH, DEFAULT_HEIGHT, title),
                       encoding="utf-8")
        return out
    t = [float(e.time_s) for e in rul_history]
    c = [float(e.tte_hours_central) for e in rul_history]
    lo = [float(e.tte_hours_lower) for e in rul_history]
    hi = [float(e.tte_hours_upper) for e in rul_history]
    svg = _line_chart_svg(
        title=title, x_label="time (s)", y_label="RUL (hours)",
        x_vals=t,
        y_lines=[("central", c, "tab:green")],
        y_band=(lo, hi, "tab:green"),
    )
    out.write_text(svg, encoding="utf-8")
    return out


def plot_actual_vs_predicted(
    eval_pairs: Sequence[Tuple[float, float]],
    path: Path,
    *,
    title: str = "Actual vs predicted RUL",
) -> Path:
    """Scatter of ``(truth, central_pred)`` pairs with y=x (SVG)."""
    out = _ensure_path(path)
    if not eval_pairs:
        out.write_text(_empty_svg(550.0, 550.0, title), encoding="utf-8")
        return out
    truth = [float(p[0]) for p in eval_pairs]
    pred = [float(p[1]) for p in eval_pairs]
    svg = _scatter_svg(
        title=title,
        x_label="actual RUL (hours)",
        y_label="predicted RUL (hours)",
        x_vals=truth, y_vals=pred, ref_line=True,
    )
    out.write_text(svg, encoding="utf-8")
    return out


def plot_prediction_uncertainty(
    eval_pairs: Sequence[Tuple[float, float, float, float]],
    path: Path,
    *,
    title: str = "RUL prediction uncertainty",
) -> Path:
    """Error-bar plot of ``(lower, central, upper)`` per sample (SVG)."""
    out = _ensure_path(path)
    if not eval_pairs:
        out.write_text(_empty_svg(DEFAULT_WIDTH, DEFAULT_HEIGHT, title),
                       encoding="utf-8")
        return out
    sorted_pairs = sorted(
        ((float(t), float(l), float(c), float(u)) for (t, l, c, u) in eval_pairs),
        key=lambda p: p[2],
    )
    truth = [p[0] for p in sorted_pairs]
    lo = [p[1] for p in sorted_pairs]
    central = [p[2] for p in sorted_pairs]
    hi = [p[3] for p in sorted_pairs]
    svg = _errorbar_svg(
        title=title,
        x_label="sample (sorted by predicted central)",
        y_label="RUL (hours)",
        central=central, lower=lo, upper=hi, truth=truth,
    )
    out.write_text(svg, encoding="utf-8")
    return out


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "plot_actual_vs_predicted",
    "plot_health_vs_time",
    "plot_prediction_uncertainty",
    "plot_rul_vs_time",
]
