from typing import List, Dict, Any
from collections import deque
from langgraph.graph import StateGraph, END
from core.state import GraphState, Query, Task, Cluster, Category
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
        self.llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
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
            return {"clusters_list": []} # Nothing to review

        # Convert Pydantic models to dicts for prompt creation
        finalized_categories_dicts = [cat.model_dump() for cat in finalized_categories]
        clusters_to_review_dicts = [cluster.model_dump() for cluster in clusters_to_review]

        prompt = create_review_prompt(finalized_categories_dicts, clusters_to_review_dicts)
        print("Calling LLM for review...")
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o", # Or your preferred LLM
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Your task is to review unlabeled query clusters and output your decisions in a structured XML format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            llm_output = response.choices[0].message.content
            print("LLM Output received.")
            # print(llm_output) # For debugging

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

        except Exception as e:
            print(f"Error during LLM review or XML parsing: {e}")
            # Handle error: e.g., retry, or mark clusters for manual review
            for cluster in clusters_to_review:
                cluster.judge = {"action": "subdivide", "k_value": 2} # Default to subdivide on error

        return {"clusters_list": clusters_to_review}

    def dispatcher_node(self, state: GraphState) -> Dict[str, Any]:
        print("---Entering dispatcher_node---")
        clusters_to_process = state["clusters_list"]
        current_tasks = state["tasks"]
        current_categories = state["categories"]

        for cluster in clusters_to_process:
            decision = cluster.judge
            if not decision:
                print(f"Warning: No decision for cluster {cluster.id}. Skipping.")
                continue

            action = decision["action"]
            print(f"Processing cluster {cluster.id} with action: {action}")

            if action == "create":
                description = decision["description"]
                create_new_category_tool(state, cluster, description)
            elif action == "assign":
                target_id = decision["target_id"]
                assign_to_existing_tool(state, cluster, target_id)
            elif action == "subdivide":
                new_k = decision["k_value"]
                subdivide_task_tool(state, cluster, new_k)
            else:
                print(f"Unknown action: {action} for cluster {cluster.id}")
        
        # Clear clusters_list after processing
        return {"tasks": current_tasks, "categories": current_categories, "clusters_list": []}

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
        
        # Ensure categories is a dict, not a list, for direct assignment
        if not isinstance(initial_state["categories"], dict):
            initial_state["categories"] = {}

        for s in self.graph.stream(initial_state):
            print(s)
            print("\n---")
        return self.graph.invoke(initial_state)
