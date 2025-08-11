from typing import List, Dict, Deque, Any, Optional, TypedDict
from collections import deque
from pydantic import BaseModel, Field

class Query(BaseModel):
    id: str
    content: str
    embedding: List[float]
    def __str__(self):
        return f"Query(id={self.id}, content={self.content})"

class Task(BaseModel):
    queries: List[Query]
    k_value: int

class Cluster(BaseModel):
    id: str
    queries: List[Query]
    samples: List[str] = Field(default_factory=list)
    judge: Optional[Dict[str, Any]] = None
    def __str__(self):
        return f"Cluster(id={self.id}, queries_count={len(self.queries)})"

class Category(BaseModel):
    id: str
    description: str
    queries: List[Query]
    samples: List[str]
    def __str__(self):
        return f"Category(id={self.id}, queries_count={len(self.queries)})"

# LangGraph的全局状态定义
class CategoryNode(BaseModel):
    id: str
    description: str
    children: List['CategoryNode'] = Field(default_factory=list)

class GraphState(TypedDict):
    tasks: Deque[Task]
    categories: Dict[str, Category]
    clusters_list: Optional[List[Cluster]]
    category_tree: Dict[str, CategoryNode]  # 新增的树状结构
