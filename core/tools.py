"""
核心工具函数模块
包含所有用于操作GraphState的内部工具函数
"""
import uuid
from typing import List, Dict, Any, Optional
from collections import deque

from .state import GraphState, Category, Task, Query, Cluster


def create_new_category_tool(state: GraphState, clusters: List[Cluster], description: str):
    """
    创建新类别，支持多个cluster合并创建
    
    Args:
        state: 全局状态对象
        clusters: 支持单个或多个cluster输入
        description: 由LLM生成的Rich Description，包含具体例子
    """
    # 生成新的类别ID
    new_category_id = f"CAT-{len(state['categories']) + 1:03d}"
    
    # 合并所有clusters的queries
    all_queries = []
    all_samples = []
    
    for cluster in clusters:
        all_queries.extend(cluster.queries)
        all_samples.extend(cluster.samples)
    
    # 创建新类别
    from config.config_loader import CONFIG
    max_samples = CONFIG.get('clustering', {}).get('max_samples_per_cluster', 10)
    
    new_category = Category(
        id=new_category_id,
        description=description,
        queries=all_queries,
        samples=all_samples[:max_samples]  # 使用配置的样本数量
    )
    
    # 添加到状态中
    state['categories'][new_category_id] = new_category
    
    print(f"创建新类别 {new_category_id}，包含 {len(all_queries)} 个查询")


def assign_to_existing_tool(state: GraphState, cluster: Cluster, target_id: str, 
                          description_update: str = "no_update"):
    """
    分配cluster到现有类别，可选地更新类别描述
    
    Args:
        state: 全局状态对象
        cluster: 要分配的cluster
        target_id: 目标类别ID
        description_update: "no_update" 或新的描述文本
    """
    if target_id not in state['categories']:
        print(f"错误: 目标类别 {target_id} 不存在，这不应该发生（validation应该已检查）")
        return
    
    existing_category = state['categories'][target_id]
    
    # 合并queries
    existing_category.queries.extend(cluster.queries)
    existing_category.samples.extend(cluster.samples)
    
    # 更新description（如果需要）
    if description_update != "no_update":
        existing_category.description = description_update
        print(f"已更新类别 {target_id} 的描述")
    else:
        print(f"类别 {target_id} 描述保持不变")
    
    print(f"分配cluster {cluster.id} 到类别 {target_id}，现包含 {len(existing_category.queries)} 个查询")


def subdivide_task_tool(state: GraphState, cluster: Cluster, new_k: int, 
                       min_cluster_size: int):
    """
    创建细分任务并加入队列
    
    Args:
        state: 全局状态对象
        cluster: 要细分的cluster
        new_k: 新的k值
        min_cluster_size: 最小簇大小（从全局状态获取）
    """
    
    # 检查最小簇尺寸 - 如果cluster已经很小还要subdivide，说明内容混乱，归为垃圾类
    if len(cluster.queries) < min_cluster_size:
        print(f"Cluster {cluster.id} 尺寸过小 ({len(cluster.queries)} < {min_cluster_size}) 却仍需subdivide，"
              f"说明内容语义混乱，归入垃圾类别")
        
        # 确保垃圾类别存在
        if "TRASH_CATEGORY" not in state['categories']:
            state['categories']["TRASH_CATEGORY"] = Category(
                id="TRASH_CATEGORY",
                description="混乱语义簇 - 包含语义混杂、无法进一步归类的查询\n典型例子：各类零散的、主题不明确的查询",
                queries=[],
                samples=[]
            )
            state['categories']["TRASH_CATEGORY"].samples.extend(cluster.samples)
        
        # 分配到垃圾类别
        state['categories']["TRASH_CATEGORY"].queries.extend(cluster.queries)

        return
    
    # 创建新的细分任务
    new_task = Task(
        queries=cluster.queries,
        k_value=new_k
    )
    
    state['tasks'].append(new_task)
    print(f"创建细分任务：{len(cluster.queries)} 个查询，k={new_task.k_value}")


def validate_decisions(clusters: List[Cluster], decisions: List[Dict[str, Any]]) -> tuple[bool, str]:
    """
    验证决策完整性：确保每个cluster都被判断且无重复
    
    Args:
        clusters: 待判断的clusters列表
        decisions: LLM返回的决策列表
    
    Returns:
        tuple[bool, str]: (是否有效, 错误信息)
    """
    cluster_ids = {c.id for c in clusters}
    decision_ids = set()
    
    # 收集决策中涉及的所有cluster ID
    for decision in decisions:
        ids_str = decision.get("id", "")
        if not ids_str:
            return False, f"决策缺少id字段: {decision}"
        
        # 支持逗号分隔的多个ID
        ids = [id.strip() for id in ids_str.split(",")]
        for cluster_id in ids:
            if cluster_id in decision_ids:
                return False, f"重复的cluster ID: {cluster_id}"
            decision_ids.add(cluster_id)
    
    # 检查是否完全匹配
    missing_ids = cluster_ids - decision_ids
    extra_ids = decision_ids - cluster_ids
    
    if missing_ids:
        return False, f"遗漏的cluster IDs: {missing_ids}"
    if extra_ids:
        return False, f"多余的cluster IDs: {extra_ids}"
    
    return True, "决策验证通过"



def parse_cluster_ids(ids_str: str) -> List[str]:
    """
    解析逗号分隔的cluster ID字符串
    
    Args:
        ids_str: 如 "temp-a,temp-b" 或 "temp-c"
    
    Returns:
        List[str]: cluster ID列表
    """
    if not ids_str:
        return []
    return [id.strip() for id in ids_str.split(",") if id.strip()]


def get_clusters_by_ids(clusters: List[Cluster], cluster_ids: List[str]) -> List[Cluster]:
    """
    根据ID列表获取对应的cluster对象
    
    Args:
        clusters: 所有可用的cluster列表
        cluster_ids: 目标cluster ID列表
    
    Returns:
        List[Cluster]: 匹配的cluster对象列表
    """
    id_to_cluster = {c.id: c for c in clusters}
    result = []
    
    for cluster_id in cluster_ids:
        if cluster_id in id_to_cluster:
            result.append(id_to_cluster[cluster_id])
        else:
            print(f"警告: 找不到cluster {cluster_id}")
    
    return result