"""
LangGraph流程编排模块
实现TDR聚类的完整工作流：clusterer -> reviewer -> dispatcher
"""
from typing import List, Dict, Any
from collections import deque
from langgraph.graph import StateGraph, END

from .state import GraphState, Query, Task, Cluster
from .tools import (
    create_new_category_tool, 
    assign_to_existing_tool, 
    subdivide_task_tool,
    parse_cluster_ids,
    get_clusters_by_ids
)
from .prompts import (
    create_cluster_display_dict,
    create_category_display_dict
)
from services.embedding_service import EmbeddingService
from services.clustering_service import ClusteringService


class TDRClusterGraph:
    """TDR聚类主工作流"""
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.embedding_service = EmbeddingService()
        self.clustering_service = ClusteringService()
        
        # 构建LangGraph工作流
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(GraphState)
        
        # 添加节点
        workflow.add_node("clusterer", self.clusterer_node)
        workflow.add_node("reviewer", self.reviewer_node)
        workflow.add_node("dispatcher", self.dispatcher_node)
        
        # 设置流程
        workflow.set_entry_point("clusterer")
        workflow.add_edge("clusterer", "reviewer")
        workflow.add_edge("reviewer", "dispatcher")
        
        # 条件循环
        workflow.add_conditional_edges(
            "dispatcher",
            self._should_continue,
            {
                "continue": "clusterer",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def clusterer_node(self, state: GraphState) -> Dict[str, Any]:
        """
        聚类节点：从队列取任务，执行K-Means聚类
        """
        print("🔵 进入clusterer节点")
        
        tasks = state["tasks"]
        if not tasks:
            print("   队列为空，跳过聚类")
            return {"clusters_list": []}
        
        # 从队列取出任务
        current_task = tasks.popleft()
        queries_to_cluster = current_task.queries
        k_value = current_task.k_value
        
        print(f"   处理任务: {len(queries_to_cluster)} 个查询 -> {k_value} 个clusters")
        
        # 执行聚类
        clusters = self.clustering_service.perform_clustering(queries_to_cluster, k_value)
        
        print(f"   聚类完成: 生成 {len(clusters)} 个有效clusters")
        
        return {
            "tasks": tasks,
            "clusters_list": clusters
        }
    
    def reviewer_node(self, state: GraphState) -> Dict[str, Any]:
        """
        评审节点：调用LLM对clusters进行决策
        """
        print("🟡 进入reviewer节点")
        
        clusters_to_review = state.get("clusters_list", [])
        categories = state.get("categories", {})
        
        if not clusters_to_review:
            print("   无clusters需要评审")
            return {"clusters_list": []}
        
        print(f"   评审 {len(clusters_to_review)} 个clusters")
        
        # 准备LLM输入数据
        finalized_categories = [
            create_category_display_dict(cat) for cat in categories.values()
        ]
        clusters_for_review = [
            create_cluster_display_dict(cluster) for cluster in clusters_to_review
        ]
        
        # 使用缓存的总查询数（避免重复计算）
        total_queries = state["total_queries"]
        
        # 使用LLM服务的高级方法进行cluster分析（包含重试机制）
        try:
            validation_result = self.llm_service.analyze_clusters_with_retry(
                finalized_categories, 
                clusters_for_review,
                state.get("dataset_name")
            )
            
            # 将决策附加到对应的clusters
            decisions = validation_result['decisions']
            print(f"🔍 调试：准备附加 {len(decisions)} 个决策到 {len(clusters_to_review)} 个clusters")
            for i, decision in enumerate(decisions):
                print(f"   决策 {i+1}: ID={decision.get('id')}, Action={decision.get('action')}")
            self._attach_decisions_to_clusters(clusters_to_review, decisions)
            
        except Exception as e:
            print(f"❌ LLM分析失败: {str(e)}")
            raise
        
        return {
            "tasks": state.get("tasks", deque()),
            "categories": state.get("categories", {}),
            "clusters_list": clusters_to_review
        }
    
    def dispatcher_node(self, state: GraphState) -> Dict[str, Any]:
        """
        分发节点：根据LLM决策执行相应操作
        """
        print("🟢 进入dispatcher节点")
        
        clusters_to_process = state.get("clusters_list", [])
        
        if not clusters_to_process:
            print("   无clusters需要处理")
            return {
                "tasks": state.get("tasks", deque()),
                "categories": state.get("categories", {}),
                "clusters_list": []
            }
        
        print(f"   处理 {len(clusters_to_process)} 个clusters的决策")
        
        # 使用缓存的统计信息（避免重复计算）
        total_queries = state["total_queries"]
        min_cluster_size = state["min_cluster_size"]
        
        for cluster in clusters_to_process:
            decision = cluster.judge
            if not decision:
                print(f"⚠️ Cluster {cluster.id} 无决策，跳过")
                continue
            
            action = decision.get("action", "").lower()
            print(f"   处理 {cluster.id}: {action}")
            
            try:
                if action == "create":
                    self._handle_create_action(state, cluster, decision)
                elif action == "assign":
                    self._handle_assign_action(state, cluster, decision)
                elif action == "subdivide":
                    self._handle_subdivide_action(state, cluster, decision, min_cluster_size)
                else:
                    print(f"❌ 未知动作: {action}")
            except Exception as e:
                print(f"❌ 处理cluster {cluster.id} 时出错: {str(e)}")
        
        return {
            "tasks": state.get("tasks", deque()),
            "categories": state.get("categories", {}),
            "clusters_list": []  # 清空已处理的clusters
        }
    
    
    def _attach_decisions_to_clusters(self, clusters: List[Cluster], decisions: List[Dict[str, Any]]):
        """将LLM决策附加到对应的clusters"""
        cluster_dict = {cluster.id: cluster for cluster in clusters}
        print(f"🔍 调试：可用cluster IDs: {list(cluster_dict.keys())}")
        
        attached_count = 0
        for decision in decisions:
            cluster_ids = parse_cluster_ids(decision.get("id", ""))
            print(f"🔍 调试：解析决策ID '{decision.get('id')}' -> {cluster_ids}")
            
            for cluster_id in cluster_ids:
                if cluster_id in cluster_dict:
                    cluster_dict[cluster_id].judge = decision
                    attached_count += 1
                    print(f"   ✓ 附加决策到 {cluster_id}: {decision.get('action')}")
                else:
                    print(f"   ❌ 找不到cluster: {cluster_id}")
        
        print(f"🔍 调试：总共附加了 {attached_count} 个决策")
    
    def _handle_create_action(self, state: GraphState, cluster: Cluster, decision: Dict[str, Any]):
        """处理create动作（支持多cluster合并）"""
        cluster_ids = parse_cluster_ids(decision.get("id", ""))
        all_clusters = get_clusters_by_ids(state.get("clusters_list", []), cluster_ids)
        
        if not all_clusters:
            print(f"⚠️ 找不到clusters: {cluster_ids}")
            return
        
        description = decision.get("description", "新创建的类别")
        create_new_category_tool(state, all_clusters, description)
    
    def _handle_assign_action(self, state: GraphState, cluster: Cluster, decision: Dict[str, Any]):
        """处理assign动作（支持多cluster合并assign）"""
        cluster_ids = parse_cluster_ids(decision.get("id", ""))
        all_clusters = get_clusters_by_ids(state.get("clusters_list", []), cluster_ids)
        
        if not all_clusters:
            print(f"⚠️ 找不到clusters: {cluster_ids}")
            return
        
        target_id = decision.get("target_id")
        description_update = decision.get("description_update", "no_update")
        
        if not target_id:
            print(f"❌ assign动作缺少target_id")
            return
        
        # 合并所有clusters到目标category
        for cluster_item in all_clusters:
            assign_to_existing_tool(state, cluster_item, target_id, "no_update")
        
        # 最后统一更新description
        if description_update != "no_update" and target_id in state['categories']:
            state['categories'][target_id].description = description_update
            print(f"已更新类别 {target_id} 的描述")
    
    def _handle_subdivide_action(self, state: GraphState, cluster: Cluster, 
                                decision: Dict[str, Any], min_cluster_size: int):
        """处理subdivide动作"""
        k_value = decision.get("k_value")
        if not k_value:
            print(f"❌ subdivide动作缺少k_value")
            return
        subdivide_task_tool(state, cluster, k_value, min_cluster_size)
    
    
    def _should_continue(self, state: GraphState) -> str:
        """判断是否继续循环"""
        tasks = state.get("tasks", deque())
        if len(tasks) > 0:
            print(f"   队列还有 {len(tasks)} 个任务，继续循环")
            return "continue"
        else:
            print("   队列为空，结束流程")
            return "end"
    
    def run(self, initial_queries: List[Query], initial_k: int = 10, dataset_name: str = None) -> Dict[str, Any]:
        """
        运行完整的TDR聚类流程
        
        Args:
            initial_queries: 初始查询列表
            initial_k: 初始聚类数
            dataset_name: 数据集名称，用于获取特定的high-level目标
            
        Returns:
            Dict[str, Any]: 最终状态，包含所有categories
        """
        print(f"🚀 开始TDR聚类流程")
        print(f"   初始数据: {len(initial_queries)} 个查询")
        print(f"   初始K值: {initial_k}")
        
        # 计算全局统计信息（一次性计算，避免重复）
        total_queries = len(initial_queries)
        min_cluster_size = self.clustering_service.calculate_min_cluster_size(total_queries)
        
        print(f"📊 初始统计信息:")
        print(f"   总查询数: {total_queries}")
        print(f"   最小簇大小: {min_cluster_size}")
        
        # 构建初始状态
        initial_task = Task(queries=initial_queries, k_value=initial_k)
        initial_state: GraphState = {
            "tasks": deque([initial_task]),
            "categories": {},
            "clusters_list": [],
            "dataset_name": dataset_name,
            "total_queries": total_queries,
            "min_cluster_size": min_cluster_size
        }
        
        # 执行图工作流
        from config.config_loader import CONFIG
        recursion_limit = CONFIG.get('system', {}).get('recursion_limit', 100)
        
        final_state = self.graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
        print("✅ TDR聚类流程完成")
        print(f"   最终类别数: {len(final_state.get('categories', {}))}")
        
        return final_state