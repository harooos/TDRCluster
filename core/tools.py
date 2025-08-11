import uuid
from .state import GraphState, Category, Task, Query, Cluster
from typing import Dict
from graphviz import Digraph
import json
from pathlib import Path
from .state import CategoryNode

def create_new_category_tool(state: GraphState, cluster: Cluster, description: str):
    new_category_id = f"CAT-{len(state['categories']) + 1:03d}"
    new_category = Category(
        id=new_category_id,
        description=description,
        queries=cluster.queries,
        samples=cluster.samples
    )
    state['categories'][new_category_id] = new_category
    
def assign_to_existing_tool(state: GraphState, cluster: Cluster, target_id: str):
    if target_id in state['categories']:
        existing_category = state['categories'][target_id]
        existing_category.queries.extend(cluster.queries)
        # For simplicity, just extend samples. A more sophisticated approach might resample.
        existing_category.samples.extend(cluster.samples)
        # Update description - this might need a more advanced LLM call to refine
        # For now, we'll just note that it's an optimization point.
        # existing_category.description = "Updated description based on new queries"
    else:
        print(f"Warning: Target category {target_id} not found for assignment.")
    
def subdivide_task_tool(state: GraphState, cluster: Cluster, new_k: int):
    new_task = Task(
        queries=cluster.queries,
        k_value=new_k
    )
    state['tasks'].append(new_task)

def visualize_category_tree(category_tree: Dict[str, CategoryNode], output_dir: str = "output"):
    """可视化类别树结构并保存为图片和JSON"""
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 保存为JSON文件
    json_path = Path(output_dir) / "category_tree.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(category_tree, f, indent=2, ensure_ascii=False, default=lambda o: o.__dict__)
    
    # 2. 使用graphviz可视化
    dot = Digraph(comment='Category Tree', format='png')
    dot.attr('node', shape='box', style='rounded')
    
    # 添加所有节点
    for node_id, node in category_tree.items():
        dot.node(node_id, f"{node_id}\n{node.description}")
    
    # 添加边关系
    for node_id, node in category_tree.items():
        for child in node.children:
            dot.edge(node_id, child.id)
    
    # 保存图片
    image_path = Path(output_dir) / "category_tree"
    dot.render(image_path, cleanup=True)
    
    print(f"树结构已保存到: {json_path} 和 {image_path}.png")
