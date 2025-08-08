import uuid
from .state import GraphState, Category, Task, Query

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
