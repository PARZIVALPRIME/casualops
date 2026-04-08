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
        if len(parts) == 2:
            src, tgt = parts[0].strip(), parts[1].strip()
            
            # Prevent double-counting the same correct hypothesis
            hyp_key = f"{src}->{tgt}"
            already_claimed = any(f"{h.cause}->{h.effect}" == hyp_key for h in agent.hypotheses[:-1])
            
            if not already_claimed:
                # Check if it's a real edge and NOT a planted phantom trap
                is_real = dag.has_edge(src, tgt)
                is_trap = any(p.correlated_node == src and p.phantom_node == tgt for p in traps.phantoms)
                
                if is_real and not is_trap:
                    comp.hypothesis_quality = 0.30
                    explanation.append("Correct new causal hypothesis (+0.30)")
                else:
                    comp.hypothesis_quality = 0.0
                    explanation.append("Incorrect or phantom causal hypothesis (+0.0)")
            else:
                explanation.append("Duplicate hypothesis ignored (+0.0)")
    
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
        # Default passive reward for no pending pressure
        comp.communication_quality = 0.10

    # 5. Time Pressure Management (0.0 to 0.15)
    # Simple check: are they taking action while they have time?
    time_ratio = time_remaining_s / max(time_budget_s, 1.0)
    comp.time_pressure_management = min(0.15, time_ratio * 0.15)

    # Clamp total to strictly (0, 1) exclusive range
    clamped_total = round(max(0.001, min(0.999, comp.total)), 3)

    return Reward(
        components=comp,
        total=clamped_total,
        explanation=" | ".join(explanation) if explanation else "Standard step"
    )
