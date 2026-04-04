"""Dense intermediate reward calculation logic"""
from causality import CausalDAG, ScenarioTraps
from models import AgentState, Reward, RewardComponents

def compute_step_reward(
    agent: AgentState,
    dag: CausalDAG,
    traps: ScenarioTraps,
    step: int,
    max_steps: int,
    time_remaining_s: float,
    time_budget_s: float,
    action_type: str,
    action_target: str,
    pending_stakeholder_responses: int,
    responded_this_step: bool,
) -> Reward:
    
    comp = RewardComponents()
    explanation = []

    # 1. Hypothesis Quality (0.0 to 0.30)
    if action_type == "hypothesize":
        parts = action_target.split("->")
        if len(parts) == 2 and dag.has_edge(parts[0].strip(), parts[1].strip()):
            comp.hypothesis_quality = 0.30
            explanation.append("Correct causal hypothesis (+0.30)")
        else:
            comp.hypothesis_quality = 0.0
            explanation.append("Incorrect causal hypothesis (+0.0)")
    
    # 2. Information Efficiency (0.0 to 0.20)
    if action_type == "observe":
        if agent.total_observation_cost > 0:
            efficiency = agent.useful_observations / max(len(agent.observations_made), 1)
            comp.information_efficiency = efficiency * 0.20
            explanation.append(f"Observation efficiency {efficiency:.2f} (+{comp.information_efficiency:.2f})")

    # 3. Phantom Resistance (0.0 to 0.20)
    # Penalize if they are investigating phantoms
    if agent.observations_made:
        phantom_ratio = agent.phantom_investigations / max(len(agent.observations_made), 1)
        resistance = 1.0 - phantom_ratio
        comp.phantom_resistance = resistance * 0.20
        if phantom_ratio > 0:
            explanation.append(f"Phantom penalty applied (-{phantom_ratio*0.20:.2f})")
        else:
            explanation.append(f"Resisted phantoms (+{comp.phantom_resistance:.2f})")
    else:
        comp.phantom_resistance = 0.20

    # 4. Communication Quality (0.0 to 0.15)
    if action_type == "communicate" and responded_this_step:
        comp.communication_quality = 0.15
        explanation.append("Stakeholder update provided (+0.15)")
    elif pending_stakeholder_responses > 0:
        comp.communication_quality = 0.0
        explanation.append("Stakeholders waiting for response (+0.0)")
    else:
        comp.communication_quality = 0.15

    # 5. Time Pressure Management (0.0 to 0.15)
    # Simple check: are they taking action while they have time?
    time_ratio = time_remaining_s / max(time_budget_s, 1.0)
    comp.time_pressure_management = min(0.15, time_ratio * 0.15)

    return Reward(
        components=comp,
        total=comp.total,
        explanation=" | ".join(explanation) if explanation else "Standard step"
    )
