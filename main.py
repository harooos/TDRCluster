#!/usr/bin/env python3
"""
TDRCluster主启动文件
"""
import json
from pathlib import Path
from datetime import datetime

from config.config_loader import CONFIG
from core.graph import TDRClusterGraph
from services.dataset_manager import DatasetManager
from services.llm_service import LLMService


def load_params() -> dict:
    """加载运行参数"""
    runtime_config = CONFIG.get("runtime", {})
    return {
        "dataset": runtime_config.get("dataset", "banking77"),
        "sample_size": runtime_config.get("sample_size", None),
        "output_dir": "output",
    }


def save_results(final_state: dict, output_dir: str, dataset_name: str) -> tuple[Path, Path]:
    """保存聚类结果"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"💾 保存结果到: {output_dir}/")

    import pandas as pd

    categories = final_state.get("categories", {})
    residual_clusters = final_state.get("residual_clusters", [])

    results_data = []
    for category in categories.values():
        for query in category.queries:
            results_data.append(
                {
                    "query_id": query.id,
                    "query_content": query.content,
                    "category_id": category.id,
                    "category_name": category.name,
                    "category_definition": category.definition,
                    "boundary_notes": category.boundary_notes,
                    "dataset": dataset_name,
                    "timestamp": timestamp,
                }
            )

    csv_path = None
    if results_data:
        results_df = pd.DataFrame(results_data)
        csv_path = output_path / f"{dataset_name}_clustering_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"   ✓ 详细结果: {csv_path}")

    summary_data = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "total_queries": sum(len(cat.queries) for cat in categories.values()),
        "total_categories": len(categories),
        "residual_cluster_count": len(residual_clusters),
        "residual_query_count": sum(len(cluster.queries) for cluster in residual_clusters),
        "categories": {},
        "residual_clusters": {},
    }

    for cat_id, category in categories.items():
        summary_data["categories"][cat_id] = {
            "name": category.name,
            "definition": category.definition,
            "boundary_notes": category.boundary_notes,
            "query_count": len(category.queries),
            "positive_examples": category.positive_examples[:5],
        }

    for cluster in residual_clusters[:20]:
        summary_data["residual_clusters"][cluster.id] = {
            "query_count": len(cluster.queries),
            "depth": cluster.depth,
            "is_noise": cluster.is_noise,
            "sample_texts": cluster.sample_texts[:5],
            "reason": (cluster.judge or {}).get("residual_reason", "unspecified"),
        }

    json_path = output_path / f"{dataset_name}_summary_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, ensure_ascii=False, indent=2)
    print(f"   ✓ 摘要报告: {json_path}")

    return csv_path, json_path


def print_results_summary(final_state: dict, dataset_name: str):
    """打印结果摘要"""
    categories = final_state.get("categories", {})
    tasks_remaining = final_state.get("tasks", [])
    residual_clusters = final_state.get("residual_clusters", [])

    print("\n" + "=" * 70)
    print(f"📊 TDRCluster V2结果摘要 - {dataset_name}")
    print("=" * 70)

    print(f"最终类别数: {len(categories)}")
    print(f"残差簇数: {len(residual_clusters)}")
    print(f"剩余任务数: {len(tasks_remaining)}")

    total_queries = sum(len(cat.queries) for cat in categories.values())
    residual_queries = sum(len(cluster.queries) for cluster in residual_clusters)
    print(f"总分类查询数: {total_queries}")
    print(f"残差查询数: {residual_queries}")

    if categories:
        print("\n📁 类别详情:")
        for index, (cat_id, category) in enumerate(categories.items(), 1):
            query_count = len(category.queries)
            print(f"\n{index}. {cat_id} | {category.name} ({query_count} 个查询)")
            print(f"   定义: {category.definition}")
            if category.boundary_notes:
                print(f"   边界: {category.boundary_notes}")
            if category.positive_examples:
                print("   样本:")
                for sample_index, sample in enumerate(category.positive_examples[:3], 1):
                    print(f"     {sample_index}. {sample[:80]}...")

    print("=" * 70)


def main():
    """主函数"""
    print("🚀 TDRCluster - LLM驱动的动态意图聚类系统\n")

    print("⚙️ 初始化LLM服务...")
    llm_service = LLMService(CONFIG.get("llm", {}))
    print("✅ LLM服务初始化完成\n")

    params = load_params()

    print("⚙️ 运行参数:")
    print(f"   数据集: {params['dataset']}")
    print("   物理切分: UMAP + HDBSCAN")
    if params["sample_size"]:
        print(f"   采样大小: {params['sample_size']}")
    print(f"   输出目录: {params['output_dir']}")
    print()

    print("🔍 初始化数据集管理器...")
    dataset_manager = DatasetManager()

    try:
        queries = dataset_manager.load_dataset_as_queries(
            params["dataset"],
            params["sample_size"],
        )
        if not queries:
            print("❌ 没有加载到任何查询数据")
            return
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    print("🔧 初始化TDR聚类系统...")
    try:
        tdr_graph = TDRClusterGraph(llm_service)
        print("   ✓ TDR系统初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return

    print("\n⚡ 开始TDR聚类流程...")
    print("   这可能需要几分钟时间，请耐心等待...")
    print("=" * 70)

    try:
        final_state = tdr_graph.run(
            initial_queries=queries,
        )
        print("=" * 70)
        print("✅ TDR聚类流程完成!")
    except Exception as e:
        print(f"❌ 聚类流程失败: {e}")
        return

    print_results_summary(final_state, params["dataset"])

    try:
        csv_path, json_path = save_results(
            final_state,
            params["output_dir"],
            params["dataset"],
        )
        print("\n💾 结果已保存:")
        if csv_path:
            print(f"   详细数据: {csv_path}")
        print(f"   摘要报告: {json_path}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

    print("\n🎉 TDRCluster聚类任务完成!")


if __name__ == "__main__":
    main()
