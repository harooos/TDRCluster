#!/usr/bin/env python3
"""
Generate a lightweight README-friendly SVG from a summary JSON and hierarchy JSON.
"""
from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an SVG snapshot for README examples.")
    parser.add_argument("summary_json", type=str, help="Path to a summary JSON file.")
    parser.add_argument("hierarchy_json", type=str, help="Path to a hierarchy JSON file.")
    parser.add_argument("output_svg", type=str, help="Path to the output SVG file.")
    return parser.parse_args()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3].rstrip()}..."


def walk_leaf_ids(node: Dict[str, Any]) -> List[str]:
    if "leaf_id" in node:
        return [node["leaf_id"]]

    leaf_ids: List[str] = []
    for child in node.get("children", []):
        leaf_ids.extend(walk_leaf_ids(child))
    return leaf_ids


def rect(x: int, y: int, width: int, height: int, fill: str, stroke: str = "none", radius: int = 18) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" />'
    )


def text(x: int, y: int, value: str, size: int = 16, weight: int = 400, fill: str = "#122033") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}">{escape(value)}</text>'
    )


def build_svg(summary: Dict[str, Any], hierarchy: Dict[str, Any]) -> str:
    width = 1200
    height = 860
    bg = "#f5f7fb"
    panel = "#ffffff"
    border = "#d8dfeb"
    muted = "#5d6b82"
    dark = "#122033"
    accent = "#0f6fff"
    accent_soft = "#dcebff"
    palette = ["#dcebff", "#d9f6eb", "#fff0d6", "#f7e4ff", "#ffe3e1"]

    categories = summary.get("categories", {})
    category_counts = {
        category_id: int(category["query_count"])
        for category_id, category in categories.items()
    }
    top_categories = sorted(
        categories.items(),
        key=lambda item: int(item[1]["query_count"]),
        reverse=True,
    )[:8]

    root_summaries = []
    for index, root in enumerate(hierarchy.get("roots", [])):
        leaf_ids = walk_leaf_ids(root)
        total_queries = sum(category_counts[leaf_id] for leaf_id in leaf_ids)
        root_summaries.append(
            {
                "name": root.get("name", root.get("leaf_id", "Leaf")),
                "leaf_count": len(leaf_ids),
                "query_count": total_queries,
                "fill": palette[index % len(palette)],
            }
        )

    total_queries = int(summary.get("total_queries", 0))
    total_categories = int(summary.get("total_categories", 0))
    residual_query_count = int(summary.get("residual_query_count", 0))
    root_count = int(hierarchy.get("root_count", len(root_summaries)))
    max_depth = int(hierarchy.get("max_depth", 0))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{bg}" />',
        text(52, 68, "TDRCluster: banking77 example", size=30, weight=700),
        text(
            52,
            100,
            "A README snapshot of the final flat taxonomy and its post-hoc hierarchy.",
            size=16,
            fill=muted,
        ),
    ]

    stat_specs = [
        ("Leaf Categories", str(total_categories)),
        ("Root Groups", str(root_count)),
        ("Classified Queries", f"{total_queries:,}"),
        ("Residual Queries", f"{residual_query_count:,}"),
        ("Hierarchy Depth", str(max_depth)),
    ]

    stat_x = 52
    for label, value in stat_specs:
        parts.append(rect(stat_x, 132, 204, 92, panel, border, 16))
        parts.append(text(stat_x + 18, 168, label, size=14, weight=500, fill=muted))
        parts.append(text(stat_x + 18, 205, value, size=30, weight=700, fill=dark))
        stat_x += 218

    parts.append(rect(52, 258, 470, 540, panel, border, 22))
    parts.append(text(80, 302, "Top-level hierarchy groups", size=22, weight=700))
    parts.append(
        text(
            80,
            330,
            "Each box aggregates one root node in the generated browsing tree.",
            size=14,
            fill=muted,
        )
    )

    root_y = 360
    for root in root_summaries:
        parts.append(rect(80, root_y, 414, 74, root["fill"], "none", 16))
        parts.append(text(100, root_y + 30, truncate(root["name"], 34), size=19, weight=700))
        parts.append(
            text(
                100,
                root_y + 56,
                f'{root["leaf_count"]} leaf categories | {root["query_count"]:,} queries',
                size=14,
                fill=muted,
            )
        )
        root_y += 90

    parts.append(rect(558, 258, 590, 540, panel, border, 22))
    parts.append(text(586, 302, "Largest leaf categories", size=22, weight=700))
    parts.append(
        text(
            586,
            330,
            "Top leaf nodes by assigned query count in the final flat taxonomy.",
            size=14,
            fill=muted,
        )
    )

    chart_left = 586
    chart_top = 378
    chart_right = 1102
    chart_width = chart_right - chart_left
    bar_max_width = 250
    row_height = 50
    max_count = max(int(category["query_count"]) for _, category in top_categories)

    for row_index, (category_id, category) in enumerate(top_categories):
        y = chart_top + row_index * row_height
        count = int(category["query_count"])
        bar_width = int(bar_max_width * count / max_count)
        label = truncate(category["name"], 44)

        parts.append(text(chart_left, y, category_id, size=13, weight=700, fill=accent))
        parts.append(text(chart_left + 74, y, label, size=15, weight=500, fill=dark))
        parts.append(rect(chart_left + 74, y + 12, bar_max_width, 12, "#edf2fa", "none", 6))
        parts.append(rect(chart_left + 74, y + 12, max(6, bar_width), 12, accent, "none", 6))
        parts.append(text(chart_left + 74 + bar_max_width + 18, y + 24, str(count), size=14, weight=700))

    parts.append(rect(558, 720, 590, 78, accent_soft, "none", 18))
    parts.append(text(586, 754, "Leaf taxonomy stays unchanged during hierarchy building.", size=18, weight=700))
    parts.append(
        text(
            586,
            780,
            "The hierarchy is a separate browsing layer built after the flat taxonomy converges.",
            size=14,
            fill=muted,
        )
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    args = parse_args()
    summary_path = Path(args.summary_json)
    hierarchy_path = Path(args.hierarchy_json)
    output_path = Path(args.output_svg)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(summary, hierarchy), encoding="utf-8")
    print(f"SVG written to {output_path}")


if __name__ == "__main__":
    main()
