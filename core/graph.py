from typing import List, Dict, Any
from collections import deque
from langgraph.graph import StateGraph, END
from core.state import GraphState, Query, Task, Cluster, Category,CategoryNode
from core.tools import create_new_category_tool, assign_to_existing_tool, subdivide_task_tool
from core.prompts import create_review_prompt
from services.embedding_service import EmbeddingService
from services.clustering_service import ClusteringService
from config.settings import settings
import xml.etree.ElementTree as ET
from openai import OpenAI

class TDRClusterGraph:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.clustering_service = ClusteringService()
        self.llm_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("clusterer", self.clusterer_node)
        workflow.add_node("reviewer", self.reviewer_node)
        workflow.add_node("dispatcher", self.dispatcher_node)

        workflow.set_entry_point("clusterer")

        workflow.add_edge("clusterer", "reviewer")
        workflow.add_edge("reviewer", "dispatcher")

        workflow.add_conditional_edges(
            "dispatcher",
            self.should_continue,
            {
                "continue": "clusterer",
                "end": END
            }
        )

        return workflow.compile()

    def clusterer_node(self, state: GraphState) -> Dict[str, Any]:
        print("---Entering clusterer_node---")
        tasks = state["tasks"]
        if not tasks:
            return {"clusters_list": []} # No tasks, nothing to cluster

        current_task = tasks.popleft() # Get the next task
        queries_to_cluster = current_task.queries
        k_value = current_task.k_value

        clusters = self.clustering_service.perform_clustering(queries_to_cluster, k_value)
        print(f"Clustered {len(queries_to_cluster)} queries into {len(clusters)} clusters.")
        return {"tasks": tasks, "clusters_list": clusters}

    def reviewer_node(self, state: GraphState) -> Dict[str, Any]:
        print("---Entering reviewer_node---")
        clusters_to_review = state["clusters_list"]
        finalized_categories = list(state["categories"].values())

        if not clusters_to_review:
            print("No clusters to review.")
            return {"clusters_list": []}

        # 保持原有LLM评审逻辑，不再处理小簇
        finalized_categories_dicts = [cat.model_dump() for cat in finalized_categories]
        clusters_to_review_dicts = [cluster.model_dump() for cluster in clusters_to_review]

        prompt = create_review_prompt(finalized_categories_dicts, clusters_to_review_dicts)
        print("Calling LLM for review...")
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst..."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            llm_output = response.choices[0].message.content
            print("LLM Output received.")
            print(f"Raw LLM output: {llm_output}")  # Add debug output
            
            try:
                # Clean XML response if needed
                llm_output = llm_output.strip()
                if not llm_output.startswith('<'):
                    llm_output = llm_output[llm_output.find('<'):]
                
                # Parse XML output
                root = ET.fromstring(llm_output)
                decisions = []
                
                for decision_elem in root.findall('decision'):
                    decision = {
                        "id": decision_elem.find('id').text,
                        "action": decision_elem.find('action').text
                    }
                    if decision["action"] == "create":
                        decision["description"] = decision_elem.find('description').text
                    elif decision["action"] == "assign":
                        decision["target_id"] = decision_elem.find('target_id').text
                    elif decision["action"] == "subdivide":
                        decision["k_value"] = int(decision_elem.find('k_value').text)
                    decisions.append(decision)
                
                # Attach decisions to clusters
                for cluster in clusters_to_review:
                    for d in decisions:
                        if d["id"] == cluster.id:
                            cluster.judge = d
                            break
            
            except ET.ParseError as e:
                print(f"XML parsing failed. Raw output:\n{llm_output}")
                raise ValueError(f"Invalid XML format from LLM: {str(e)}") from e

        except Exception as e:
            print(f"Error during LLM review: {str(e)}")
            # Fallback behavior
            for cluster in clusters_to_review:
                cluster.judge = {
                    "action": "subdivide",
                    "k_value": min(5, len(cluster.queries)),
                    "error": str(e)
                }

        return {"clusters_list": clusters_to_review}

    def dispatcher_node(self, state: GraphState) -> Dict[str, Any]:
        print("---Entering dispatcher_node---")
        clusters_to_process = state["clusters_list"]
        current_tasks = state["tasks"]
        current_categories = state["categories"]
        
        # 初始化树状结构(如果不存在)
        if "category_tree" not in state:
            state["category_tree"] = {}
    
        # 确保垃圾簇存在
        if "TRASH_CATEGORY" not in current_categories:
            current_categories["TRASH_CATEGORY"] = Category(
                id="TRASH_CATEGORY",
                description="垃圾簇 - 包含查询数过少的簇",
                queries=[],
                samples=[]
            )
            state["category_tree"]["TRASH_CATEGORY"] = CategoryNode(
                id="TRASH_CATEGORY",
                description="垃圾簇 - 包含查询数过少的簇"
            )
    
        for cluster in clusters_to_process:
            decision = cluster.judge
            if not decision:
                print(f"Warning: No decision for cluster {cluster.id}. Skipping.")
                continue
    
            action = decision["action"]
            print(f"Processing cluster {cluster.id} with action: {action}")
    
            # 处理小簇逻辑
            if action in ["create", "subdivide"] and len(cluster.queries) < 3:
                print(f"Redirecting small cluster {cluster.id} to TRASH_CATEGORY (only {len(cluster.queries)} queries)")
                assign_to_existing_tool(state, cluster, "TRASH_CATEGORY")
                continue
    
            if action == "create":
                # 创建新类别并添加到树结构
                new_category_id = f"CAT-{len(state['categories']) + 1:03d}"
                state["category_tree"][new_category_id] = CategoryNode(
                    id=new_category_id,
                    description=decision["description"],
                    children=[]
                )
                create_new_category_tool(state, cluster, decision["description"])
                
            elif action == "assign":
                # 分配到现有类别不需要修改树结构
                assign_to_existing_tool(state, cluster, decision["target_id"])
                
            elif action == "subdivide":
                # 记录细分操作的父子关系
                parent_id = cluster.id
                new_k = decision["k_value"]
                
                # 确保父节点存在于树中
                if parent_id not in state["category_tree"]:
                    state["category_tree"][parent_id] = CategoryNode(
                        id=parent_id,
                        description=f"Parent cluster {parent_id}",
                        children=[]
                    )
                
                # 执行细分操作
                subdivide_task_tool(state, cluster, new_k)
                
                # 为每个子簇创建节点并添加到父节点
                for i in range(new_k):
                    child_id = f"{parent_id}-{i+1}"
                    state["category_tree"][child_id] = CategoryNode(
                        id=child_id,
                        description=f"Subdivided from {parent_id}",
                        children=[]
                    )
                    state["category_tree"][parent_id].children.append(
                        state["category_tree"][child_id]
                    )
            else:
                print(f"Unknown action: {action} for cluster {cluster.id}")
        
        return {"tasks": current_tasks, "categories": current_categories, 
                "clusters_list": [], "category_tree": state["category_tree"]}

    def should_continue(self, state: GraphState) -> str:
        if state["tasks"]:
            print("Tasks queue is not empty. Continuing...")
            return "continue"
        else:
            print("Tasks queue is empty. Ending graph.")
            return "end"

    def run(self, initial_queries: List[Query], initial_k: int):
        initial_task = Task(queries=initial_queries, k_value=initial_k)
        initial_state = GraphState(tasks=deque([initial_task]), categories={}, clusters_list=[])
        
        if not isinstance(initial_state["categories"], dict):
            initial_state["categories"] = {}
    
        # 增加递归限制配置
        config = {
            "recursion_limit": 1000,  # 提高到100次
            "debug": True  # 启用调试输出
        }
    
        def _print_state(state):
            print("Current State:")
            print(f"Pending Tasks: {len(state.get('tasks', deque()))}")
            print(f"Current Clusters: {len(state.get('clusters_list', []))}")
            print(f"Categories Count: {len(state.get('categories', {}))}")
            print("---")
    
        for s in self.graph.stream(initial_state, config):  # 传入配置
            _print_state(s)
        return self.graph.invoke(initial_state)
