#!/usr/bin/env python3
"""Plot study context and cache curves as dependency-free SVG.

Reads `result.json` files from `<results-dir>/*/` and emits hand-written SVGs
(no matplotlib, no external fonts/images) suitable for embedding in Markdown
reports.

Three figures are produced:
  1. prefill-vs-context   median prefill tok/s against context fill (percent
                          of ctx), one series per configured profile;
  2. decode-vs-context    median decode tok/s against context fill, same;
  3. ab-cache-hit         cache hit fraction in A/B order for `ab-sequence`
                          runs, one curve per `(profile, order)` so different
                          orders never overwrite each other.

Ladder series come from the same per-step median aggregation as the report
(METHODOLOGY.md section 11.1): `mode == "ladder"`, `status == "ok"`,
`completion_is_fixed == true`, aggregated per `target_pct`; the 0% warm-up step
is excluded. Profiles with role `control` are drawn dashed. When the config file
is absent, profiles, title and A/B orders are auto-detected from the raw runs;
raw profile names are never disclosed: each series is shown as a neutral
placeholder (`profile N`) and the missing-data warning reports only a count. No
profile names, labels or model names are hard-coded.

Usage from the case root:
    python3 scripts/report/plot_context_curves.py --lang ru
    python3 .../plot_context_curves.py --lang en --out-dir docs/figures

`--case-dir` and `--repo-root` configure the absolute path prefixes scrubbed
from captions and warnings, exactly as in `generate_report.py`.
"""

import argparse
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_report as report  # noqa: E402  (sibling module, stdlib only)

WIDTH = 1000
HEIGHT = 600

MARGIN_LEFT = 95
MARGIN_RIGHT = 35
MARGIN_TOP = 75
MARGIN_BOTTOM = 75

GRID_COLOR = "#dddddd"
AXIS_COLOR = "#333333"
TEXT_COLOR = "#222222"
FONT = "sans-serif"

SERIES_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#ff7f0e", "#8c564b", "#17becf", "#e377c2",
]
SESSION_COLORS = {"A": "#1f77b4", "B": "#d62728"}
CONTROL_DASH = "6,4"

LABELS = {
    "ru": {
        "prefill": {
            "title": "Prefill против заполнения контекста",
            "xlabel": "Заполненность контекста, % от ctx",
            "ylabel": "Скорость prefill, tok/s",
            "points": "точек",
            "no_data_for": "нет данных по",
        },
        "decode": {
            "title": "Decode против заполнения контекста",
            "xlabel": "Заполненность контекста, % от ctx",
            "ylabel": "Скорость decode, tok/s",
            "points": "точек",
            "no_data_for": "нет данных по",
        },
        "ab": {
            "title": "A/B-кеш: доля попаданий по порядку обращений",
            "xlabel": "Порядок обращений",
            "ylabel": "Доля попаданий в кеш",
            "session_a": "сессия A",
            "session_b": "сессия B",
            "note": "порядки показаны раздельно; cache-hit от рантайма не зависит",
        },
        "canonical_note": "серии — медианный агрегат fixed-прогонов (completion_is_fixed)",
        "limited": "limited",
        "points": "точек",
        "no_data_for": "нет данных по",
        "no_data_count": "нет данных по профилям: {n}",
        "profile": "профиль",
    },
    "en": {
        "prefill": {
            "title": "Prefill vs context fill",
            "xlabel": "Context fill, % of ctx",
            "ylabel": "Prefill speed, tok/s",
            "points": "points",
            "no_data_for": "no data for",
        },
        "decode": {
            "title": "Decode vs context fill",
            "xlabel": "Context fill, % of ctx",
            "ylabel": "Decode speed, tok/s",
            "points": "points",
            "no_data_for": "no data for",
        },
        "ab": {
            "title": "A/B cache: hit fraction by request order",
            "xlabel": "Request order",
            "ylabel": "Cache hit fraction",
            "session_a": "session A",
            "session_b": "session B",
            "note": "orders are shown separately; cache-hit does not depend on the runtime",
        },
        "canonical_note": "series use the median aggregate of fixed runs (completion_is_fixed)",
        "limited": "limited",
        "points": "points",
        "no_data_for": "no data for",
        "no_data_count": "no data for {n} profile(s)",
        "profile": "profile",
    },
}

MARKER_SHAPES = ("circle", "square", "triangle", "diamond")
DASH_STYLES = (None, "9,5", "3,3", "12,4,2,4")


def study_title(cfg, lang):
    return report.scrub_text(
        cfg["title"].get(lang) or cfg["title"].get("ru") or ""
    )


def fmt_number(value, digits=2):
    return f"{value:.{digits}f}"


def series_from_aggregated(aggregation, cfg, metric):
    """One median curve per configured profile from its ladder aggregate.

    The X axis is the requested context fill (`target_pct`) and the Y value is
    the per-step median of `metric`. Returns `{profile: {"points": [...],
    "n": int, "limited": bool, "run_ids": [...]}}`.
    """
    series = {}
    for name in report.profile_names(cfg):
        aggregate = aggregation.get(name)
        if not aggregate:
            continue
        points = []
        for step in aggregate["steps"]:
            stat = step["metrics"].get(metric)
            if not stat:
                continue
            points.append((float(step["target_pct"]), float(stat["median"])))
        if points:
            points.sort(key=lambda item: item[0])
            series[name] = {
                "points": points,
                "n": aggregate["n"],
                "limited": aggregate["limited"],
                "run_ids": aggregate["run_ids"],
            }
    return series


def ab_series(runs):
    """Cache-hit curves keyed by `(profile, order)` so both A/B orders survive."""
    series = {}
    for run in runs:
        if run["mode"] != "ab-sequence":
            continue
        name = run["profile"] or run["run_id"] or "unknown"
        order = run["order"] or "?"
        points = series.setdefault((name, order), {"A": [], "B": []})
        for step in run["steps"] or []:
            order_index = step.get("order_index")
            session = step.get("session")
            value = step.get("cache_hit_fraction")
            if session in points and report.is_number(order_index) and report.is_number(value):
                points[session].append((float(order_index), float(value)))
    for points in series.values():
        for session in points:
            points[session].sort(key=lambda item: item[0])
    return {key: points for key, points in series.items() if points["A"] or points["B"]}


def svg_open(title):
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'font-family="{FONT}" font-size="13">',
        f"<title>{escape(title)}</title>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
    ]


def add_text(parts, x, y, text, size=13, anchor="start", weight=None, fill=TEXT_COLOR, rotate=None):
    extra = ""
    if weight:
        extra += f' font-weight="{weight}"'
    if rotate is not None:
        extra += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" fill="{fill}"{extra}>{escape(text)}</text>'
    )


def marker_svg(shape, cx, cy, color, radius=4):
    if shape == "square":
        return (
            f'<rect x="{cx - radius:.1f}" y="{cy - radius:.1f}" '
            f'width="{2 * radius}" height="{2 * radius}" fill="{color}" '
            f'stroke="#ffffff" stroke-width="1.5"/>'
        )
    if shape == "triangle":
        points = f"{cx:.1f},{cy - radius:.1f} {cx + radius:.1f},{cy + radius:.1f} {cx - radius:.1f},{cy + radius:.1f}"
        return f'<polygon points="{points}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
    if shape == "diamond":
        points = (
            f"{cx:.1f},{cy - radius:.1f} {cx + radius:.1f},{cy:.1f} "
            f"{cx:.1f},{cy + radius:.1f} {cx - radius:.1f},{cy:.1f}"
        )
        return f'<polygon points="{points}" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" '
        f'stroke="#ffffff" stroke-width="1.5"/>'
    )


def legend_box(parts, x, y, entries, width=255):
    height = 22 * len(entries) + 14
    parts.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height}" '
        f'fill="#ffffff" fill-opacity="0.88" stroke="#999999" stroke-width="1" rx="4"/>'
    )
    for index, entry in enumerate(entries):
        item_y = y + 20 + index * 22
        dash = f' stroke-dasharray="{entry["dash"]}"' if entry.get("dash") else ""
        parts.append(
            f'<line x1="{x + 12:.1f}" y1="{item_y - 4:.1f}" '
            f'x2="{x + 52:.1f}" y2="{item_y - 4:.1f}" '
            f'stroke="{entry["color"]}" stroke-width="2.5"{dash}/>'
        )
        if entry.get("marker"):
            parts.append(marker_svg(entry["marker"], x + 52, item_y - 4, entry["color"], radius=3.5))
        add_text(parts, x + 62, item_y, entry["text"])


def nice_ticks(y_min, y_max, count=6):
    if y_max <= y_min:
        y_max = y_min + 1.0
    step = (y_max - y_min) / count
    return [y_min + step * index for index in range(count + 1)]


def configured_label_names(raw_config):
    """Profile names whose label was set explicitly in the config file.

    Auto-detected labels equal the raw profile names, so anything not listed
    here must never be rendered or printed as-is.
    """
    names = set()
    if not isinstance(raw_config, dict):
        return names
    for item in raw_config.get("profiles") or []:
        if isinstance(item, dict) and item.get("name") and item.get("label"):
            names.add(str(item["name"]))
    return names


def display_label(cfg, name, lang, configured, order):
    """Configured caption, or a generic placeholder when the config is absent."""
    if name in configured:
        return report.scrub_text(report.profile_label(cfg, name, lang))
    if name in order:
        return f"{LABELS[lang]['profile']} {order.index(name) + 1}"
    return LABELS[lang]["profile"]


def series_label(label, lang, info, point_word):
    suffix = f" [{LABELS[lang]['limited']} n={info['n']}]" if info.get("limited") else ""
    return f"{label} ({len(info['points'])} {point_word}){suffix}"


def build_curve_svg(series, labels, cfg, lang, x_max, y_max, control_names, configured, order):
    height = HEIGHT
    plot_x0 = MARGIN_LEFT
    plot_x1 = WIDTH - MARGIN_RIGHT
    plot_y0 = MARGIN_TOP
    plot_y1 = height - MARGIN_BOTTOM
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    def sx(value):
        return plot_x0 + (value / x_max) * plot_w if x_max else plot_x0

    def sy(value):
        return plot_y1 - (value / y_max) * plot_h if y_max else plot_y1

    title = f"{labels['title']} \u2014 {study_title(cfg, lang)}"
    parts = svg_open(title)
    parts.append(
        "<defs>"
        f'<clipPath id="plotClip"><rect x="{plot_x0}" y="{plot_y0}" '
        f'width="{plot_w}" height="{plot_h}"/></clipPath>'
        "</defs>"
    )

    x_tick_count = 5
    x_ticks = [x_max * index / x_tick_count for index in range(x_tick_count + 1)]
    y_ticks = nice_ticks(0.0, y_max)

    for tick in x_ticks:
        x = sx(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_y0}" x2="{x:.1f}" y2="{plot_y1}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
    for tick in y_ticks:
        y = sy(tick)
        parts.append(
            f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
    for tick in x_ticks:
        add_text(parts, sx(tick), plot_y1 + 22, f"{int(tick)}%", anchor="middle")
    for tick in y_ticks:
        add_text(parts, plot_x0 - 12, sy(tick) + 4, fmt_number(tick, digits=0), anchor="end")

    parts.append(
        f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1.5"/>'
    )
    add_text(parts, (plot_x0 + plot_x1) / 2, plot_y1 + 52, labels["xlabel"], size=15, anchor="middle")
    add_text(parts, 26, (plot_y0 + plot_y1) / 2, labels["ylabel"], size=15, anchor="middle", rotate=-90)
    add_text(parts, WIDTH / 2, 34, title, size=17, anchor="middle", weight="bold")
    add_text(parts, WIDTH / 2, 56, LABELS[lang]["canonical_note"], size=12, anchor="middle")

    entries = []
    parts.append('<g clip-path="url(#plotClip)">')
    for index, name in enumerate(sorted(series)):
        info = series[name]
        points = info["points"]
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        dash = CONTROL_DASH if name in control_names else None
        entries.append({
            "color": color,
            "text": series_label(display_label(cfg, name, lang, configured, order), lang, info, labels["points"]),
            "marker": "circle",
            "dash": dash,
        })
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr} points="{coords}"/>'
        )
        for x, y in points:
            parts.append(marker_svg("circle", sx(x), sy(y), color))
    parts.append("</g>")
    missing = [name for name in report.profile_names(cfg, "main") if name not in series]
    for name in missing:
        entries.append({
            "color": "#bbbbbb",
            "dash": "4,4",
            "marker": "circle",
            "text": f"{labels['no_data_for']} {display_label(cfg, name, lang, configured, order)}",
        })
    if entries:
        legend_box(parts, plot_x1 - 320, plot_y0 + 12, entries, width=310)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def build_ab_svg(series, labels, cfg, lang, configured, order):
    height = HEIGHT
    plot_x0 = MARGIN_LEFT
    plot_x1 = WIDTH - MARGIN_RIGHT
    plot_y0 = MARGIN_TOP
    plot_y1 = height - MARGIN_BOTTOM
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    max_index = 0
    for points in series.values():
        for session in ("A", "B"):
            for order_index, _ in points[session]:
                max_index = max(max_index, int(order_index))
    x_max = max(1, max_index)

    def sx(value):
        return plot_x0 + (value / x_max) * plot_w

    def sy(value):
        return plot_y1 - max(0.0, min(1.0, value)) * plot_h

    title = f"{labels['title']} \u2014 {study_title(cfg, lang)}"
    parts = svg_open(title)
    parts.append(
        "<defs>"
        f'<clipPath id="abClip"><rect x="{plot_x0}" y="{plot_y0}" '
        f'width="{plot_w}" height="{plot_h}"/></clipPath>'
        "</defs>"
    )
    for index in range(x_max + 1):
        x = sx(index)
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_y0}" x2="{x:.1f}" y2="{plot_y1}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        add_text(parts, x, plot_y1 + 22, str(index), anchor="middle")
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = sy(tick)
        parts.append(
            f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        add_text(parts, plot_x0 - 12, y + 4, fmt_number(tick), anchor="end")
    parts.append(
        f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" '
        f'stroke="{AXIS_COLOR}" stroke-width="1.5"/>'
    )
    add_text(parts, (plot_x0 + plot_x1) / 2, plot_y1 + 52, labels["xlabel"], size=15, anchor="middle")
    add_text(parts, 26, (plot_y0 + plot_y1) / 2, labels["ylabel"], size=15, anchor="middle", rotate=-90)
    add_text(parts, WIDTH / 2, 34, title, size=17, anchor="middle", weight="bold")
    add_text(parts, WIDTH / 2, 56, labels["note"], size=12, anchor="middle")

    series_keys = sorted(series, key=lambda key: (key[0], key[1]))
    entries = [
        {"color": SESSION_COLORS["A"], "text": labels["session_a"], "marker": "circle"},
        {"color": SESSION_COLORS["B"], "text": labels["session_b"], "marker": "circle"},
    ]
    for index, key in enumerate(series_keys):
        profile, ab_order = key
        entries.append({
            "color": "#666666",
            "dash": DASH_STYLES[index % len(DASH_STYLES)],
            "marker": MARKER_SHAPES[index % len(MARKER_SHAPES)],
            "text": f"{display_label(cfg, profile, lang, configured, order)} \u2014 {ab_order}",
        })
    parts.append('<g clip-path="url(#abClip)">')
    for index, key in enumerate(series_keys):
        points = series[key]
        dash_style = DASH_STYLES[index % len(DASH_STYLES)]
        shape = MARKER_SHAPES[index % len(MARKER_SHAPES)]
        for session in ("A", "B"):
            if not points[session]:
                continue
            color = SESSION_COLORS[session]
            dash = f' stroke-dasharray="{dash_style}"' if dash_style else ""
            coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points[session])
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
                f'stroke-linejoin="round" stroke-linecap="round"{dash} points="{coords}"/>'
            )
            for x, y in points[session]:
                parts.append(marker_svg(shape, sx(x), sy(y), color))
    parts.append("</g>")
    if entries:
        legend_box(parts, plot_x1 - 320, plot_y0 + 12, entries, width=310)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--out-dir", type=Path, default=Path("./docs/figures"))
    parser.add_argument("--config", type=Path, default=Path("./study.json"))
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--lang", choices=("ru", "en"), default="ru")
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 2

    report.configure_paths(args.results_dir, args.case_dir, args.repo_root)

    runs = report.collect_runs(args.results_dir)
    raw_config = report.load_json(args.config) if args.config and args.config.is_file() else None
    cfg = report.normalized_config(raw_config, runs)
    aggregation = report.aggregate_profiles(runs, cfg)
    labels = LABELS[args.lang]
    control_names = set(report.profile_names(cfg, "control"))
    configured = configured_label_names(raw_config)
    order = report.profile_names(cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    prefill = series_from_aggregated(aggregation, cfg, "prefill_tokens_per_second")
    decode = series_from_aggregated(aggregation, cfg, "decode_tokens_per_second")
    ab = ab_series(runs)

    missing = [name for name in report.profile_names(cfg, "main") if name not in prefill]
    unnamed = [name for name in missing if name not in configured]
    for name in missing:
        if name not in configured:
            continue
        print(f"warning: {labels['no_data_for']} {display_label(cfg, name, args.lang, configured, order)}")
    if unnamed:
        print("warning: " + labels["no_data_count"].format(n=len(unnamed)))

    x_candidates = [
        x
        for series in list(prefill.values()) + list(decode.values())
        for x, _ in series["points"]
    ]
    x_max = max([100.0] + x_candidates) * 1.05
    prefill_max = max(
        [1.0] + [y for series in prefill.values() for _, y in series["points"]]
    ) * 1.10
    decode_max = max(
        [1.0] + [y for series in decode.values() for _, y in series["points"]]
    ) * 1.10

    figures = [
        ("prefill-vs-context",
         build_curve_svg(prefill, labels["prefill"], cfg, args.lang, x_max, prefill_max,
                         control_names, configured, order),
         sum(len(series["points"]) for series in prefill.values())),
        ("decode-vs-context",
         build_curve_svg(decode, labels["decode"], cfg, args.lang, x_max, decode_max,
                         control_names, configured, order),
         sum(len(series["points"]) for series in decode.values())),
        ("ab-cache-hit", build_ab_svg(ab, labels["ab"], cfg, args.lang, configured, order),
         sum(len(points["A"]) + len(points["B"]) for points in ab.values())),
    ]
    for name, svg, count in figures:
        out = args.out_dir / f"{name}.{args.lang}.svg"
        out.write_text(svg, encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size} bytes, {count} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
