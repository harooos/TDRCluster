"""
核心工具函数模块
包含所有用于操作GraphState的内部工具函数
"""
from typing import Dict, Any, List

from config.config_loader import CONFIG

from .state import GraphState, Category, SplitTask, MicroCluster


def _max_examples() -> int:
    return CONFIG.get("clustering", {}).get("max_samples_per_cluster", 8)


def _merge_examples(existing_examples: List[str], new_examples: List[str]) -> List[str]:
    merged = []
    seen = set()

    for example in existing_examples + new_examples:
        normalized = example.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)

    return merged[:_max_examples()]


def create_new_category_tool(state: GraphState, cluster: MicroCluster, decision: Dict[str, Any]) -> str:
    """基于单个micro-cluster创建新类别"""
    new_category_id = f"CAT-{len(state['categories']) + 1:03d}"

    new_category = Category(
        id=new_category_id,
        name=decision["name"],
        definition=decision["definition"],
        boundary_notes=decision.get("boundary_notes", ""),
        positive_examples=_merge_examples([], cluster.sample_texts),
        queries=list(cluster.queries),
    )

    state["categories"][new_category_id] = new_category
    print(f"创建新类别 {new_category_id} ({new_category.name})，包含 {len(new_category.queries)} 个查询")
    return new_category_id


def merge_into_existing_category_tool(
    state: GraphState,
    cluster: MicroCluster,
    target_id: str,
    name_update: str = "no_update",
    definition_update: str = "no_update",
    boundary_update: str = "no_update",
):
    """将micro-cluster并入现有类别，并可选地更新类别边界"""
    if target_id not in state["categories"]:
        print(f"错误: 目标类别 {target_id} 不存在")
        return

    existing_category = state["categories"][target_id]
    existing_category.queries.extend(cluster.queries)
    existing_category.positive_examples = _merge_examples(
        existing_category.positive_examples,
        cluster.sample_texts,
    )

    if name_update != "no_update":
        existing_category.name = name_update
        print(f"已更新类别 {target_id} 的名称")

    if definition_update != "no_update":
        existing_category.definition = definition_update
        print(f"已更新类别 {target_id} 的定义")

    if boundary_update != "no_update":
        existing_category.boundary_notes = boundary_update
        print(f"已更新类别 {target_id} 的边界说明")

    print(f"合并 {cluster.id} -> {target_id}，现包含 {len(existing_category.queries)} 个查询")


def enqueue_resplit_task_tool(state: GraphState, cluster: MicroCluster, split_reason: str):
    """将仍不纯的micro-cluster重新塞回BFS队列做下一轮物理切分"""
    max_split_depth = state["max_split_depth"]
    min_cluster_size = state["min_cluster_size"]

    if cluster.depth >= max_split_depth:
        record_residual_cluster_tool(state, cluster, f"达到最大切分深度 {max_split_depth}")
        return

    if len(cluster.queries) <= min_cluster_size:
        record_residual_cluster_tool(
            state,
            cluster,
            f"簇规模 {len(cluster.queries)} 已不适合继续物理切分",
        )
        return

    new_task = SplitTask(
        queries=list(cluster.queries),
        depth=cluster.depth + 1,
        parent_cluster_id=cluster.id,
        split_reason=split_reason,
    )

    state["tasks"].append(new_task)
    print(
        f"创建重切分任务 {new_task.id}: {len(cluster.queries)} 个查询, "
        f"depth={new_task.depth}, reason={split_reason}"
    )


def record_residual_cluster_tool(state: GraphState, cluster: MicroCluster, reason: str):
    """将噪声或无法继续切分的簇记录到残差池中"""
    cluster.judge = {
        **(cluster.judge or {}),
        "residual_reason": reason,
    }
    state["residual_clusters"].append(cluster)
    print(f"记录残差簇 {cluster.id}: {reason}")
