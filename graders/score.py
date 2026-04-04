"""Final deterministic scoring logic (0.0 to 1.0)"""
from typing import Dict
from causality import CausalDAG, Edge, ScenarioTraps
from models import AgentState, ServiceMetrics

def compute_final_score(
    agent: AgentState,
    dag: CausalDAG,
    traps: ScenarioTraps,
    remediation_successful: bool,
    services_final: Dict[str, ServiceMetrics],
    expected_fix_effects: Dict[str, str],
    max_steps: int,
    steps_taken: int,
) -> Dict[str, float]:
    
    # 1. Causal Chain Score (0.40)
    agent_edges = set()
    for h in agent.hypotheses:
        agent_edges.add(Edge(source=h.cause, target=h.effect))
    
    true_edges = set(dag.edges)
    critical_edges = set(dag.critical_path)
    
    correct_edges = agent_edges & true_edges
    phantom_edges = agent_edges - true_edges
    
    precision = len(correct_edges) / max(len(agent_edges), 1)
    recall = len(correct_edges) / max(len(critical_edges), 1)
    phantom_penalty = len(phantom_edges) / max(len(agent_edges), 1)
    
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    causal_score = max(0.0, f1 - (0.5 * phantom_penalty)) * 0.40

    # 2. Remediation Score (0.25)
    remediation_score = 0.0
    if remediation_successful:
        # Time bonus: faster fixes get more points (up to 20% bonus of the 0.25)
        time_fraction = steps_taken / max_steps
        time_bonus = max(0.0, 1.0 - time_fraction) * 0.05
        remediation_score = 0.20 + time_bonus

    # 3. Counterfactual Score (0.15)
    counterfactual_score = 0.0
    if expected_fix_effects and agent.predictions:
        # Simple match for hackathon MVP: did they predict the right metric:service?
        expected_targets = set(expected_fix_effects.keys())
        agent_targets = set(f"{p.metric_name}:{p.service}" for p in agent.predictions)
        
        matches = expected_targets & agent_targets
        if expected_targets:
            counterfactual_score = (len(matches) / len(expected_targets)) * 0.15

    # 4. Efficiency Score (0.10)
    efficiency_score = 0.0
    if agent.total_observation_cost > 0:
        # Ideal cost is subjective, but let's assume discovering the DAG takes ~cost 5-10
        # If cost > 30, score approaches 0
        efficiency_ratio = min(1.0, 10.0 / max(agent.total_observation_cost, 1))
        efficiency_score = efficiency_ratio * 0.10

    # 5. Communication Score (0.10)
    communication_score = 0.0
    if agent.communications_sent:
        # Flat reward for attempting communication for now
        communication_score = 0.10

    total = causal_score + remediation_score + counterfactual_score + efficiency_score + communication_score

    return {
        "causal_chain": round(causal_score, 3),
        "remediation": round(remediation_score, 3),
        "counterfactual": round(counterfactual_score, 3),
        "efficiency": round(efficiency_score, 3),
        "communication": round(communication_score, 3),
        "total": round(total, 3),
    }
