import pandas as pd
from collections import deque
from core.state import Query, Task, GraphState
from core.graph import TDRClusterGraph
from services.embedding_service import EmbeddingService
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def main():
    # Initialize Embedding Service
    embedding_service = EmbeddingService()

    # 1. Load data
    # For demonstration, let's create some dummy queries. 
    # In a real scenario, you'd load from queries.csv and generate embeddings.
    # Make sure to have OPENAI_API_KEY set in your environment or .env file

    # Example: Load from CSV and generate embeddings if not already present
    queries_df = pd.read_csv("data/queries.csv")
    all_queries = []
    for index, row in queries_df.iterrows():
        query_content = row["query"]
        # Check if embedding already exists (e.g., in a cached file)
        # For simplicity, generating on the fly here. In production, cache embeddings.
        embedding = embedding_service.get_embedding(query_content)
        all_queries.append(Query(id=str(index), content=query_content, embedding=embedding))

    # 2. Initialize the graph with an initial task
    # The initial k_value can be a heuristic or user-defined
    initial_k = 5 # Example initial k

    graph_runner = TDRClusterGraph()
    final_state = graph_runner.run(initial_queries=all_queries, initial_k=initial_k)

    # 3. Output results
    print("\n--- Final Categories ---")
    for cat_id, category in final_state["categories"].items():
        print(f"Category ID: {category.id}")
        print(f"Description: {category.description}")
        print(f"Number of Queries: {len(category.queries)}")
        print(f"Sample Queries: {', '.join(category.samples)}")
        print("---")

    # Optionally, save results to a CSV or JSON file
    output_data = []
    for cat_id, category in final_state["categories"].items():
        for query in category.queries:
            output_data.append({
                "query_id": query.id,
                "query_content": query.content,
                "category_id": category.id,
                "category_description": category.description
            })
    output_df = pd.DataFrame(output_data)
    output_df.to_csv("output_categories.csv", index=False)
    print("Results saved to output_categories.csv")

if __name__ == "__main__":
    main()
