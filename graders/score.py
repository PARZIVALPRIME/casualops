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
    
    trap_edges = set(Edge(source=p.correlated_node, target=p.phantom_node) for p in traps.phantoms)
    
    true_edges = set(dag.edges) - trap_edges
    critical_edges = set(dag.critical_path)
    
    correct_edges = agent_edges & true_edges
    phantom_edges = (agent_edges - true_edges) | (agent_edges & trap_edges)
    
    precision = min(1.0, len(correct_edges) / max(len(agent_edges), 1))
    
    # Recall should only count edges that are in the critical path
    correct_critical = agent_edges & critical_edges
    recall = min(1.0, len(correct_critical) / max(len(critical_edges), 1))
    
    phantom_penalty = min(1.0, len(phantom_edges) / max(len(agent_edges), 1))
    
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    causal_score = max(0.0, f1 - (0.5 * phantom_penalty)) * 0.40

    # 2. Remediation Score (0.25)
    remediation_score = 0.0
    if remediation_successful:
        # Base points for fixing it
        remediation_score = 0.20
        # Time bonus: faster fixes get more points (up to 5% bonus)
        time_fraction = steps_taken / max_steps
        time_bonus = max(0.0, (1.0 - time_fraction) * 0.05)
        remediation_score += time_bonus
    elif any(r.startswith("restart:") or r.startswith("scale:") for r in agent.remediations_applied):
        # Small partial credit if they tried the right target but maybe wrong action or ran out of time
        # This rewards intent in the right direction
        remediation_score = 0.05

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
        # Penalize if they never responded to the VP's adversarial pressure
        # Score is ratio of sent vs baseline (minimum 1)
        communication_score = min(0.10, len(agent.communications_sent) * 0.05)

    total = causal_score + remediation_score + counterfactual_score + efficiency_score + communication_score

    # Hackathon constraint: every score must be strictly between 0 and 1
    def clamp(v: float) -> float:
        return round(max(0.001, min(0.999, v)), 3)

    return {
        "causal_chain": clamp(causal_score),
        "remediation": clamp(remediation_score),
        "counterfactual": clamp(counterfactual_score),
        "efficiency": clamp(efficiency_score),
        "communication": clamp(communication_score),
        "total": clamp(total),
    }
