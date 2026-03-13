"""
LangGraph流程编排模块
实现TDRCluster V2的完整工作流：splitter -> reviewer -> dispatcher
"""
import re
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import deque

from langgraph.graph import StateGraph, END

from config.config_loader import CONFIG

from .state import GraphState, Query, SplitTask, MicroCluster
from .tools import (
    create_new_category_tool,
    merge_into_existing_category_tool,
    enqueue_resplit_task_tool,
    record_residual_cluster_tool,
)
from .prompts import create_cluster_display_dict, create_category_display_dict
from services.clustering_service import ClusteringService
from services.run_control import CancelCallback, check_cancelled


class TDRClusterGraph:
    """TDRCluster V2主工作流"""

    def __init__(
        self,
        llm_service,
        observer: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        should_cancel: CancelCallback = None,
        checkpoint_callback: Optional[Callable[[GraphState], None]] = None,
    ):
        self.llm_service = llm_service
        self.clustering_service = ClusteringService()
        self.inspect_config = CONFIG.get("clustering", {}).get("inspect", {})
        self.observer = observer
        self.should_cancel = should_cancel
        self.checkpoint_callback = checkpoint_callback
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
        check_cancelled(self.should_cancel)
        print("🔵 进入splitter节点")

        existing_clusters = state.get("clusters_list", [])
        if existing_clusters:
            print(f"   当前仍有 {len(existing_clusters)} 个micro-cluster待评审，跳过切分")
            next_state = self._build_next_state(
                state,
                {
                    "tasks": state.get("tasks", deque()),
                    "categories": state.get("categories", {}),
                    "clusters_list": existing_clusters,
                    "current_cluster": None,
                    "residual_clusters": state.get("residual_clusters", []),
                },
            )
            self._checkpoint(next_state)
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
            next_state = self._build_next_state(
                state,
                {
                    "tasks": tasks,
                    "categories": state.get("categories", {}),
                    "clusters_list": [],
                    "current_cluster": None,
                    "residual_clusters": state.get("residual_clusters", []),
                },
            )
            self._checkpoint(next_state)
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
        check_cancelled(self.should_cancel)

        all_residual_clusters = list(state.get("residual_clusters", []))
        all_residual_clusters.extend(residual_clusters)

        print(f"   切分完成: review={len(review_clusters)}, residual={len(residual_clusters)}")
        self._emit(
            "splitter_completed",
            task_id=current_task.id,
            depth=current_task.depth,
            review_cluster_count=len(review_clusters),
            residual_cluster_count=len(residual_clusters),
            pending_tasks=len(tasks),
        )
        next_state = self._build_next_state(
            state,
            {
                "tasks": tasks,
                "categories": state.get("categories", {}),
                "clusters_list": review_clusters,
                "current_cluster": None,
                "residual_clusters": all_residual_clusters,
            },
        )
        self._checkpoint(next_state)
        return {
            "tasks": tasks,
            "categories": state.get("categories", {}),
            "clusters_list": review_clusters,
            "current_cluster": None,
            "residual_clusters": all_residual_clusters,
        }

    def reviewer_node(self, state: GraphState) -> Dict[str, Any]:
        """评审节点：一次只调用LLM评审一个micro-cluster"""
        check_cancelled(self.should_cancel)
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
            self._emit(
                "reviewer_started",
                cluster_id=current_cluster.id,
                query_count=len(current_cluster.queries),
                remaining_microclusters=len(remaining_clusters),
                category_count=len(categories),
            )
            current_cluster.judge = self.llm_service.analyze_cluster_with_retry(
                finalized_categories,
                create_cluster_display_dict(current_cluster),
            )
            check_cancelled(self.should_cancel)
            print(f"   ✓ {current_cluster.id} 决策: {current_cluster.judge.get('action')}")
            self._emit(
                "reviewer_completed",
                cluster_id=current_cluster.id,
                action=current_cluster.judge.get("action"),
                category_count=len(categories),
            )
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
        check_cancelled(self.should_cancel)
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
            elif action == "inspect":
                updated_clusters = self._handle_inspect_request(
                    state,
                    current_cluster,
                    decision.get("reason", "need_more_evidence"),
                )
                next_state = self._build_next_state(
                    state,
                    {
                        "tasks": state.get("tasks", deque()),
                        "categories": state.get("categories", {}),
                        "clusters_list": updated_clusters,
                        "current_cluster": None,
                        "residual_clusters": state.get("residual_clusters", []),
                    },
                )
                self._checkpoint(next_state)
                return {
                    "tasks": state.get("tasks", deque()),
                    "categories": state.get("categories", {}),
                    "clusters_list": updated_clusters,
                    "current_cluster": None,
                    "residual_clusters": state.get("residual_clusters", []),
                }
            else:
                record_residual_cluster_tool(state, current_cluster, f"未知动作: {action}")
        except Exception as e:
            record_residual_cluster_tool(state, current_cluster, f"处理失败: {str(e)}")

        self._emit(
            "dispatcher_completed",
            cluster_id=current_cluster.id,
            action=action,
            category_count=len(state.get("categories", {})),
            residual_count=len(state.get("residual_clusters", [])),
            pending_tasks=len(state.get("tasks", deque())),
            pending_microclusters=len(state.get("clusters_list", [])),
        )
        next_state = self._build_next_state(
            state,
            {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": state.get("clusters_list", []),
                "current_cluster": None,
                "residual_clusters": state.get("residual_clusters", []),
            },
        )
        self._checkpoint(next_state)

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

    def _handle_inspect_request(
        self,
        state: GraphState,
        cluster: MicroCluster,
        inspect_reason: str,
    ) -> List[MicroCluster]:
        check_cancelled(self.should_cancel)
        normalized_reason = inspect_reason.strip() or "need_more_evidence"
        previous_reasons = list(cluster.inspect_reasons)
        additional_evidence = self.clustering_service.sample_additional_evidence(cluster)

        if additional_evidence.get("query_ids"):
            cluster.add_evidence_round(
                round_index=len(cluster.evidence_rounds) + 1,
                mode="inspect",
                groups=additional_evidence.get("groups", {}),
                query_ids=additional_evidence.get("query_ids", []),
            )

        next_inspect_round = cluster.inspect_round + 1
        should_force_final, stop_reason = self._should_force_final_review(
            inspect_reason=normalized_reason,
            previous_reasons=previous_reasons,
            additional_evidence=additional_evidence,
            next_inspect_round=next_inspect_round,
        )

        cluster.inspect_round = next_inspect_round
        cluster.inspect_reasons.append(normalized_reason)
        cluster.allow_inspect = not should_force_final
        cluster.inspect_stop_reason = stop_reason if should_force_final else None
        cluster.judge = None

        if should_force_final:
            print(
                f"   {cluster.id} inspect停止，进入最终裁决: {stop_reason} "
                f"(novelty_ratio={additional_evidence.get('novelty_ratio', 0.0):.2f}, "
                f"relative_novelty={additional_evidence.get('relative_novelty', 0.0):.2f}, "
                f"coverage={additional_evidence.get('coverage_ratio', 0.0):.2%})"
            )
        else:
            print(
                f"   {cluster.id} 补充证据后继续评审: "
                f"新增样本={len(additional_evidence.get('query_ids', []))}, "
                f"novelty_ratio={additional_evidence.get('novelty_ratio', 0.0):.2f}, "
                f"relative_novelty={additional_evidence.get('relative_novelty', 0.0):.2f}, "
                f"coverage={additional_evidence.get('coverage_ratio', 0.0):.2%}"
            )

        self._emit(
            "inspect_completed",
            cluster_id=cluster.id,
            next_round=cluster.inspect_round,
            added_samples=len(additional_evidence.get("query_ids", [])),
            novelty_ratio=float(additional_evidence.get("novelty_ratio", 0.0)),
            coverage_ratio=float(additional_evidence.get("coverage_ratio", 0.0)),
            forced_final=should_force_final,
            stop_reason=stop_reason,
        )

        return [cluster] + list(state.get("clusters_list", []))

    def _should_force_final_review(
        self,
        *,
        inspect_reason: str,
        previous_reasons: List[str],
        additional_evidence: Dict[str, Any],
        next_inspect_round: int,
    ) -> Tuple[bool, str]:
        max_rounds = int(self.inspect_config.get("max_rounds", 4))
        min_novelty_ratio = float(self.inspect_config.get("min_novelty_ratio", 0.25))
        coverage_threshold = float(self.inspect_config.get("force_decision_coverage", 0.85))
        max_same_reason_repeats = int(self.inspect_config.get("max_same_reason_repeats", 2))

        new_query_ids = additional_evidence.get("query_ids", [])
        novelty_ratio = float(additional_evidence.get("novelty_ratio", 0.0))
        relative_novelty = float(additional_evidence.get("relative_novelty", 0.0))
        coverage_ratio = float(additional_evidence.get("coverage_ratio", 0.0))

        if not new_query_ids:
            return True, "no_more_unseen_evidence"

        if next_inspect_round > max_rounds:
            return True, f"safety_fuse_max_rounds={max_rounds}"

        if novelty_ratio < min_novelty_ratio:
            return True, f"low_novelty_ratio={novelty_ratio:.2f}"

        if coverage_ratio >= coverage_threshold:
            return True, f"high_coverage={coverage_ratio:.2%}"

        normalized_reason = self._normalize_reason(inspect_reason)
        repeated_reason_count = 1 + sum(
            1
            for reason in previous_reasons
            if self._normalize_reason(reason) == normalized_reason
        )
        if (
            normalized_reason
            and repeated_reason_count >= max_same_reason_repeats
            and novelty_ratio < 0.5
        ):
            return True, (
                "repeated_reason_without_new_signal="
                f"{repeated_reason_count}:{novelty_ratio:.2f}:{relative_novelty:.2f}"
            )

        return False, ""

    def _normalize_reason(self, reason: str) -> str:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", reason.lower())
        return " ".join(normalized.split())

    def run(
        self,
        initial_queries: Optional[List[Query]] = None,
        initial_state: Optional[GraphState] = None,
    ) -> Dict[str, Any]:
        """运行完整的TDRCluster V2流程"""
        check_cancelled(self.should_cancel)
        print("🚀 开始TDRCluster V2流程")
        if initial_state is None and initial_queries is None:
            raise ValueError("initial_queries 和 initial_state 不能同时为空")

        if initial_state is None:
            print(f"   初始数据: {len(initial_queries)} 个查询")
        else:
            print(f"   从断点恢复，待处理任务数: {len(initial_state.get('tasks', deque()))}")
        print("   物理切分引擎: UMAP + HDBSCAN")
        working_state = initial_state

        if working_state is None:
            total_queries = len(initial_queries)
            min_cluster_size = self.clustering_service.calculate_min_cluster_size(total_queries)
            max_split_depth = CONFIG.get("clustering", {}).get("max_split_depth", 4)

            initial_task = SplitTask(
                queries=initial_queries,
                depth=0,
                parent_cluster_id=None,
                split_reason="root",
            )
            working_state = {
                "tasks": deque([initial_task]),
                "categories": {},
                "clusters_list": [],
                "current_cluster": None,
                "residual_clusters": [],
                "total_queries": total_queries,
                "min_cluster_size": min_cluster_size,
                "max_split_depth": max_split_depth,
            }

        self._emit("run_started", total_queries=working_state["total_queries"])
        self._checkpoint(working_state)

        recursion_limit = CONFIG.get("system", {}).get("recursion_limit", 100)
        final_state = self.graph.invoke(working_state, config={"recursion_limit": recursion_limit})
        check_cancelled(self.should_cancel)

        print("✅ TDRCluster V2流程完成")
        print(f"   最终类别数: {len(final_state.get('categories', {}))}")
        print(f"   残差簇数: {len(final_state.get('residual_clusters', []))}")
        self._emit(
            "run_finished",
            category_count=len(final_state.get("categories", {})),
            residual_count=len(final_state.get("residual_clusters", [])),
        )
        return final_state

    def _emit(self, event: str, **payload):
        if self.observer:
            self.observer(event, payload)

    def _checkpoint(self, state: GraphState):
        if self.checkpoint_callback:
            self.checkpoint_callback(state)

    def _build_next_state(
        self,
        state: GraphState,
        updates: Dict[str, Any],
    ) -> GraphState:
        return {
            "tasks": updates.get("tasks", state.get("tasks", deque())),
            "categories": updates.get("categories", state.get("categories", {})),
            "clusters_list": updates.get("clusters_list", state.get("clusters_list", [])),
            "current_cluster": updates.get("current_cluster", state.get("current_cluster")),
            "residual_clusters": updates.get("residual_clusters", state.get("residual_clusters", [])),
            "total_queries": state["total_queries"],
            "min_cluster_size": state["min_cluster_size"],
            "max_split_depth": state["max_split_depth"],
        }
