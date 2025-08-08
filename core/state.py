from typing import List, Dict, Deque, Any, Optional, TypedDict
from collections import deque
from pydantic import BaseModel, Field

class Query(BaseModel):
    id: str
    content: str
    embedding: List[float]

class Task(BaseModel):
    queries: List[Query]
    k_value: int

class Cluster(BaseModel):
    id: str
    queries: List[Query]
    samples: List[str] = Field(default_factory=list)
    judge: Optional[Dict[str, Any]] = None

class Category(BaseModel):
    id: str
    description: str
    queries: List[Query]
    samples: List[str]

# LangGraph的全局状态定义
class GraphState(TypedDict):
    tasks: Deque[Task]
    categories: Dict[str, Category]
    clusters_list: Optional[List[Cluster]]
