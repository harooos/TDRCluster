"""
LangGraph流程编排模块
实现TDRCluster V2的完整工作流：splitter -> reviewer -> dispatcher
"""
from typing import List, Dict, Any
from collections import deque

from langgraph.graph import StateGraph, END

from config.config_loader import CONFIG

from .state import GraphState, Query, SplitTask
from .tools import (
    create_new_category_tool,
    merge_into_existing_category_tool,
    enqueue_resplit_task_tool,
    record_residual_cluster_tool,
)
from .prompts import create_cluster_display_dict, create_category_display_dict
from services.clustering_service import ClusteringService


class TDRClusterGraph:
    """TDRCluster V2主工作流"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.clustering_service = ClusteringService()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("splitter", self.splitter_node)
        workflow.add_node("reviewer", self.reviewer_node)
        workflow.add_node("dispatcher", self.dispatcher_node)

        workflow.set_entry_point("splitter")
        workflow.add_edge("splitter", "reviewer")
        workflow.add_conditional_edges(
            "dispatcher",
            self._should_continue,
            {
                "review_more": "reviewer",
                "split_more": "splitter",
                "end": END,
            },
        )
        workflow.add_edge("reviewer", "dispatcher")

        return workflow.compile()

    def splitter_node(self, state: GraphState) -> Dict[str, Any]:
        """切分节点：从BFS队列取任务，执行一次物理切分"""
        print("🔵 进入splitter节点")

        existing_clusters = state.get("clusters_list", [])
        if existing_clusters:
            print(f"   当前仍有 {len(existing_clusters)} 个micro-cluster待评审，跳过切分")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": existing_clusters,
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        tasks = state["tasks"]
        if not tasks:
            print("   队列为空，跳过切分")
            return {
                "tasks": tasks,
                "categories": state.get("categories", {}),
                "clusters_list": [],
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        current_task = tasks.popleft()
        review_clusters, residual_clusters = self.clustering_service.perform_clustering(
            current_task,
            state["total_queries"],
        )

        all_residual_clusters = list(state.get("residual_clusters", []))
        all_residual_clusters.extend(residual_clusters)

        print(f"   切分完成: review={len(review_clusters)}, residual={len(residual_clusters)}")
        return {
            "tasks": tasks,
            "categories": state.get("categories", {}),
            "clusters_list": review_clusters,
            "current_cluster": None,
            "residual_clusters": all_residual_clusters,
        }

    def reviewer_node(self, state: GraphState) -> Dict[str, Any]:
        """评审节点：一次只调用LLM评审一个micro-cluster"""
        print("🟡 进入reviewer节点")

        clusters_to_review = state.get("clusters_list", [])
        categories = state.get("categories", {})

        if not clusters_to_review:
            print("   无micro-cluster需要评审")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": categories,
                "clusters_list": [],
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        finalized_categories = [
            create_category_display_dict(category)
            for category in categories.values()
        ]
        current_cluster = clusters_to_review[0]
        remaining_clusters = clusters_to_review[1:]

        try:
            print(
                f"   评审 {current_cluster.id}: "
                f"{len(current_cluster.queries)} 个query, "
                f"剩余待审 {len(remaining_clusters)} 个micro-cluster"
            )
            current_cluster.judge = self.llm_service.analyze_cluster_with_retry(
                finalized_categories,
                create_cluster_display_dict(current_cluster),
            )
            print(f"   ✓ {current_cluster.id} 决策: {current_cluster.judge.get('action')}")
        except Exception as e:
            print(f"❌ LLM分析失败: {str(e)}")
            raise

        return {
            "tasks": state.get("tasks", deque()),
            "categories": categories,
            "clusters_list": remaining_clusters,
            "current_cluster": current_cluster,
            "residual_clusters": state.get("residual_clusters", []),
        }

    def dispatcher_node(self, state: GraphState) -> Dict[str, Any]:
        """分发节点：根据单个LLM决策更新taxonomy或继续切分"""
        print("🟢 进入dispatcher节点")

        current_cluster = state.get("current_cluster")
        if not current_cluster:
            print("   无当前micro-cluster需要处理")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": state.get("clusters_list", []),
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        decision = current_cluster.judge
        if not decision:
            record_residual_cluster_tool(state, current_cluster, "LLM未返回有效决策")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": state.get("clusters_list", []),
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            }

        action = decision.get("action", "").lower()
        print(f"   处理 {current_cluster.id}: {action}")

        try:
            if action == "create":
                create_new_category_tool(state, current_cluster, decision)
            elif action == "merge":
                merge_into_existing_category_tool(
                    state,
                    current_cluster,
                    decision["target_id"],
                    decision.get("name_update", "no_update"),
                    decision.get("definition_update", "no_update"),
                    decision.get("boundary_update", "no_update"),
                )
            elif action == "split":
                enqueue_resplit_task_tool(
                    state,
                    current_cluster,
                    decision.get("reason", "semantic_impurity"),
                )
            else:
                record_residual_cluster_tool(state, current_cluster, f"未知动作: {action}")
        except Exception as e:
            record_residual_cluster_tool(state, current_cluster, f"处理失败: {str(e)}")

        return {
            "tasks": state.get("tasks", deque()),
            "categories": state.get("categories", {}),
            "clusters_list": state.get("clusters_list", []),
            "current_cluster": None,
            "residual_clusters": state.get("residual_clusters", []),
        }

    def _should_continue(self, state: GraphState) -> str:
        clusters_to_review = state.get("clusters_list", [])
        if clusters_to_review:
            print(f"   仍有 {len(clusters_to_review)} 个micro-cluster待评审，继续review")
            return "review_more"

        tasks = state.get("tasks", deque())
        if len(tasks) > 0:
            print(f"   队列还有 {len(tasks)} 个任务，继续循环")
            return "split_more"

        print("   队列为空，结束流程")
        return "end"

    def run(self, initial_queries: List[Query]) -> Dict[str, Any]:
        """运行完整的TDRCluster V2流程"""
        print("🚀 开始TDRCluster V2流程")
        print(f"   初始数据: {len(initial_queries)} 个查询")
        print("   物理切分引擎: UMAP + HDBSCAN")

        total_queries = len(initial_queries)
        min_cluster_size = self.clustering_service.calculate_min_cluster_size(total_queries)
        max_split_depth = CONFIG.get("clustering", {}).get("max_split_depth", 4)

        initial_task = SplitTask(
            queries=initial_queries,
            depth=0,
            parent_cluster_id=None,
            split_reason="root",
        )
        initial_state: GraphState = {
            "tasks": deque([initial_task]),
            "categories": {},
            "clusters_list": [],
            "current_cluster": None,
            "residual_clusters": [],
            "total_queries": total_queries,
            "min_cluster_size": min_cluster_size,
            "max_split_depth": max_split_depth,
        }

        recursion_limit = CONFIG.get("system", {}).get("recursion_limit", 100)
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})

        print("✅ TDRCluster V2流程完成")
        print(f"   最终类别数: {len(final_state.get('categories', {}))}")
        print(f"   残差簇数: {len(final_state.get('residual_clusters', []))}")
        return final_state
