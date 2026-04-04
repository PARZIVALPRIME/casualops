"""Phantom causality and Simpson's paradox mechanics"""
from typing import List
from pydantic import BaseModel
from .graph import CausalDAG, Edge

class Phantom(BaseModel):
    phantom_node: str
    correlated_node: str

class ScenarioTraps(BaseModel):
    """Mechanics for planting false correlations and traps."""
    phantoms: List[Phantom] = []

    def inject_into_dag(self, dag: CausalDAG) -> None:
        """Inject phantom edges into the DAG.
        
        This makes them statistically correlated in the metrics, 
        but they do not belong to the critical_path causing the failure.
        """
        for p in self.phantoms:
            if p.phantom_node not in dag.nodes:
                dag.nodes.append(p.phantom_node)
            # Add a causal link from the correlated root/intermediate node 
            # to the phantom node to create the correlation.
            dag.edges.append(Edge(source=p.correlated_node, target=p.phantom_node))
