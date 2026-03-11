#!/usr/bin/env python3
"""
Build a post-hoc hierarchy from an existing flat taxonomy summary JSON.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from config.config_loader import CONFIG
from services.hierarchy_service import HierarchyService
from services.llm_service import LLMService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a semantic hierarchy from an existing summary JSON."
    )
    parser.add_argument(
        "summary_json",
        type=str,
        help="Path to a saved summary JSON, e.g. output/banking77_summary_20260311_230022.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the hierarchy JSON.",
    )
    return parser.parse_args()


def default_output_path(summary_path: Path, dataset_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return summary_path.parent / f"{dataset_name}_hierarchy_{timestamp}.json"


def main():
    args = parse_args()
    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到summary文件: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset_name = summary.get("dataset", summary_path.stem.replace("_summary", ""))

    print("⚙️ 初始化LLM服务...")
    llm_service = LLMService(CONFIG.get("llm", {}))
    hierarchy_service = HierarchyService(llm_service)

    print(f"📂 读取summary: {summary_path}")
    print(f"   数据集: {dataset_name}")
    print(f"   叶子类别数: {len(summary.get('categories', {}))}")

    hierarchy = hierarchy_service.build_hierarchy_from_summary(summary)
    hierarchy["source_summary_path"] = str(summary_path)

    output_path = Path(args.output) if args.output else default_output_path(summary_path, dataset_name)
    output_path.write_text(
        json.dumps(hierarchy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ 层级树构建完成")
    print(f"   输出文件: {output_path}")
    print(f"   root_count: {hierarchy.get('root_count')}")
    print(f"   internal_node_count: {hierarchy.get('internal_node_count')}")
    print(f"   max_depth: {hierarchy.get('max_depth')}")
    print(f"   single_child_internal_count: {hierarchy.get('single_child_internal_count')}")


if __name__ == "__main__":
    main()
