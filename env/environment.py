"""Main CausalOps Environment Logic"""
import random
from typing import Dict, List, Optional, Any

from models import (
    Observation, Action, ActionType, StepResult, State, AgentState, 
    AggregateMetrics, CounterfactualPrompt, OBSERVE_COSTS, REMEDIATE_COSTS,
    CausalClaim, CounterfactualPrediction
)
from causality.graph import CausalDAG
from causality.traps import ScenarioTraps
from tasks import TASK_REGISTRY
from graders.score import compute_final_score
from graders.reward import compute_step_reward

class CausalOpsEnvironment:
    def __init__(self):
        self._task_id: str = ""
        self._task: Any = None
        self._seed: int = 0
        self._state: State = State()
        self._dag: Optional[CausalDAG] = None
        self._traps: Optional[ScenarioTraps] = None
        self._pending_responses: int = 0

    @property
    def available_tasks(self) -> List[str]:
        return list(TASK_REGISTRY.keys())

    def reset(self, task_id: str, seed: Optional[int] = None) -> Observation:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Task {task_id} not found. Available: {self.available_tasks}")
        
        self._task_id = task_id
        self._task = TASK_REGISTRY[task_id]()
        self._seed = seed if seed is not None else random.randint(0, 99999)
        random.seed(self._seed)

        self._dag = self._task.build_dag()
        self._traps = self._task.build_traps()
        assert self._dag is not None
        assert self._traps is not None
        self._traps.inject_into_dag(self._dag)

        self._state = State(
            task_id=self._task_id,
            step_number=0,
            done=False,
            services=self._task.initial_services(),
            agent=AgentState(),
            time_elapsed_s=0.0,
            time_budget_s=self._task.time_budget_s,
            current_phase=1,
            total_reward=0.0,
            remediation_successful=False,
        )
        self._pending_responses = 0
        
        return self._build_observation(counterfactual_prompt=None)

    def _build_observation(self, counterfactual_prompt: Optional[CounterfactualPrompt]) -> Observation:
        overview = {k: v.status for k, v in self._state.services.items()}
        
        # Only include detailed metrics for observed services
        detailed_metrics = {}
        for obs in self._state.agent.observations_made:
            if obs.startswith("metrics:"):
                svc = obs.split(":")[1]
                if svc in self._state.services:
                    detailed_metrics[svc] = self._state.services[svc]
        
        # Aggregate metrics
        tot_req = sum(s.request_rate for s in self._state.services.values())
        weighted_err = sum(s.error_rate * s.request_rate for s in self._state.services.values()) / max(1, tot_req)
        avg_lat = sum(s.latency_ms for s in self._state.services.values()) / max(1, len(self._state.services))
        healthy = sum(1 for s in self._state.services.values() if s.status == "healthy")
        degraded = sum(1 for s in self._state.services.values() if s.status == "degraded")
        critical = sum(1 for s in self._state.services.values() if s.status == "critical")
        
        agg = AggregateMetrics(
            total_request_rate=tot_req,
            weighted_error_rate=weighted_err,
            avg_latency_ms=avg_lat,
            services_healthy=healthy,
            services_degraded=degraded,
            services_critical=critical
        )

        logs = []
        for obs in self._state.agent.observations_made:
            if obs.startswith("logs:"):
                svc = obs.split(":")[1]
                logs.extend(self._task.logs_for_service(svc, self._state.step_number, self._state.services))

        return Observation(
            step_number=self._state.step_number,
            time_elapsed_s=self._state.time_elapsed_s,
            time_budget_remaining_s=self._state.time_budget_s - self._state.time_elapsed_s,
            services_overview=overview,
            detailed_metrics=detailed_metrics,
            aggregate_metrics=agg,
            alerts=self._task.alerts_at_step(self._state.step_number),
            logs=logs,
            stakeholder_messages=self._task.stakeholder_messages_at_step(self._state.step_number),
            counterfactual_prompt=counterfactual_prompt,
            task_description=self._task.description,
            available_actions=[e.value for e in ActionType]
        )

    def step(self, action: Action) -> StepResult:
        if self._state.done:
            raise RuntimeError("Episode is done. Please reset().")

        assert self._dag is not None
        assert self._traps is not None

        self._state.step_number += 1
        self._state.time_elapsed_s += self._task.time_step_s
        
        # Check time budget
        if self._state.time_elapsed_s >= self._state.time_budget_s or self._state.step_number >= self._task.max_steps:
            self._state.done = True

        action_type = action.type.value
        action_target = action.target
        detail = action.detail
        
        responded_this_step = False
        prompt: Optional[CounterfactualPrompt] = None

        if action.type == ActionType.OBSERVE:
            if action_target not in self._state.agent.observations_made:
                self._state.agent.observations_made.append(action_target)
                
            obs_type = action_target.split(":")[0] if ":" in action_target else "metrics"
            cost = OBSERVE_COSTS.get(obs_type, 1)
            self._state.agent.total_observation_cost += cost
            
            # Simple heuristic for "useful" vs "phantom"
            svc = action_target.split(":")[1] if ":" in action_target else ""
            if svc in self._dag.nodes:
                is_phantom = any(p.phantom_node == svc for p in self._traps.phantoms)
                if is_phantom:
                    self._state.agent.phantom_investigations += 1
                else:
                    self._state.agent.useful_observations += 1

        elif action.type == ActionType.HYPOTHESIZE:
            parts = action_target.split("->")
            if len(parts) == 2:
                conf = 1.0
                try:
                    conf = float(detail) if detail else 1.0
                except ValueError:
                    pass
                self._state.agent.hypotheses.append(CausalClaim(cause=parts[0].strip(), effect=parts[1].strip(), confidence=conf))

        elif action.type == ActionType.REMEDIATE:
            self._state.agent.remediations_applied.append(action_target)
            cost = REMEDIATE_COSTS.get(action_target.split(":")[0] if ":" in action_target else "restart", 5)
            self._state.time_elapsed_s += cost * 5.0 # penalty in time
            
            # Prompt for prediction
            prompt = CounterfactualPrompt(
                message=f"You applied {action_target}. What metric change do you expect?",
                requires_prediction=True
            )

        elif action.type == ActionType.PREDICT:
            try:
                parts = action_target.split(":")
                metric = parts[0]
                svc = parts[1]
                det_parts = detail.split(",")
                exp_delta = float(det_parts[0])
                timeframe = float(det_parts[1]) if len(det_parts) > 1 else 30.0
                self._state.agent.predictions.append(CounterfactualPrediction(
                    metric_name=metric, service=svc, expected_delta=exp_delta, timeframe_s=timeframe, step_made=self._state.step_number
                ))
            except Exception:
                pass # ignore parsing errors

        elif action.type == ActionType.COMMUNICATE:
            self._state.agent.communications_sent.append(detail)
            responded_this_step = True
            if self._pending_responses > 0:
                self._pending_responses -= 1

        # Track pending messages
        msgs = self._task.stakeholder_messages_at_step(self._state.step_number)
        self._pending_responses += sum(1 for m in msgs if m.requires_response)

        # Evolve environment
        self._state.services = self._task.evolve(self._state.services, self._state.step_number, self._state.agent.remediations_applied)
        
        # Check if remediated properly (all services healthy)
        all_healthy = all(s.status == "healthy" for s in self._state.services.values())
        if all_healthy and self._state.agent.remediations_applied:
            self._state.remediation_successful = True
            self._state.done = True

        # Rewards
        reward = compute_step_reward(
            agent=self._state.agent,
            dag=self._dag,
            traps=self._traps,
            step=self._state.step_number,
            max_steps=self._task.max_steps,
            time_remaining_s=self._state.time_budget_s - self._state.time_elapsed_s,
            time_budget_s=self._task.time_budget_s,
            action_type=action_type,
            action_target=action_target,
            pending_stakeholder_responses=self._pending_responses,
            responded_this_step=responded_this_step
        )
        self._state.total_reward += reward.total

        obs = self._build_observation(counterfactual_prompt=prompt)
        info = {}
        
        if self._state.done:
            # Final scoring
            final_scores = compute_final_score(
                agent=self._state.agent,
                dag=self._dag,
                traps=self._traps,
                remediation_successful=self._state.remediation_successful,
                services_final=self._state.services,
                expected_fix_effects=self._task.expected_fix_effects(),
                max_steps=self._task.max_steps,
                steps_taken=self._state.step_number
            )
            info["final_scores"] = final_scores

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._state.done,
            info=info
        )

    def state(self) -> State:
        return self._state
