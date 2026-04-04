"""Ground-truth DAG logic"""
from typing import List, Set
from pydantic import BaseModel

class Edge(BaseModel):
    source: str
    target: str

    def __hash__(self):
        return hash((self.source, self.target))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.source == other.source and self.target == other.target

class CausalDAG(BaseModel):
    """Ground truth directed acyclic graph of the scenario."""
    nodes: List[str]
    edges: List[Edge]
    critical_path: List[Edge]  # The essential causal chain the agent must identify

    def has_edge(self, source: str, target: str) -> bool:
        return Edge(source=source, target=target) in self.edges
