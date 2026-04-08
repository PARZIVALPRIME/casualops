"""Main CausalOps Environment Logic — OpenEnv Environment subclass."""
import random
from typing import Dict, List, Optional, Any

from openenv.core.env_server import Environment

from models import (
    CausalOpsObservation, CausalOpsAction, ActionType, CausalOpsState, AgentState,
    AggregateMetrics, CounterfactualPrompt, OBSERVE_COSTS, REMEDIATE_COSTS,
    CausalClaim, CounterfactualPrediction, ServiceMetrics, StakeholderMessage
)
from causality.graph import CausalDAG
from causality.traps import ScenarioTraps
from tasks import TASK_REGISTRY
from graders.score import compute_final_score
from graders.reward import compute_step_reward


class CausalOpsEnvironment(Environment):
    """OpenEnv-compatible causal inference environment."""

    def __init__(self):
        super().__init__()
        self._task_id: str = ""
        self._task: Any = None
        self._seed: int = 0
        self._state: CausalOpsState = CausalOpsState()
        self._dag: Optional[CausalDAG] = None
        self._traps: Optional[ScenarioTraps] = None
        self._pending_responses: int = 0
        self._name_map: Dict[str, str] = {}
        self._reverse_map: Dict[str, str] = {}

    def _generate_names(self, seed: int) -> Dict[str, str]:
        import string
        rng = random.Random(seed)
        mapping = {}
        if self._task:
            for svc in self._task.initial_services().keys():
                suffix = ''.join(rng.choices(string.ascii_lowercase + string.digits, k=4))
                mapping[svc] = f"{svc}-{suffix}"
        return mapping

    def _translate_to_agent(self, text: str) -> str:
        for orig, mapped in self._name_map.items():
            text = text.replace(orig, mapped)
        return text

    def _translate_from_agent(self, text: str) -> str:
        for orig, mapped in self._name_map.items():
            text = text.replace(mapped, orig)
        return text

    @property
    def available_tasks(self) -> List[str]:
        return list(TASK_REGISTRY.keys())

    def reset(self, seed: Optional[int] = None, task_id: str = "easy_smoking_gun", **kwargs) -> CausalOpsObservation:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Task {task_id} not found. Available: {self.available_tasks}")

        self._task_id = task_id
        self._task = TASK_REGISTRY[task_id]()
        self._seed = seed if seed is not None else random.randint(0, 99999)
        random.seed(self._seed)

        self._name_map = self._generate_names(self._seed)
        self._reverse_map = {v: k for k, v in self._name_map.items()}

        self._dag = self._task.build_dag()
        self._traps = self._task.build_traps()
        assert self._dag is not None
        assert self._traps is not None
        self._traps.inject_into_dag(self._dag)

        self._state = CausalOpsState(
            task_id=self._task_id,
            step_number=0,
            step_count=0,
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

        return self._build_observation(counterfactual_prompt=None, reward=None, done=False)

    def _add_noise(self, val: float, scale: float) -> float:
        noisy = val + random.gauss(0, scale)
        return max(0.0, round(noisy, 2))

    def _build_observation(self, counterfactual_prompt: Optional[CounterfactualPrompt],
                           extra_messages: Optional[List[StakeholderMessage]] = None,
                           reward: Optional[float] = None,
                           done: bool = False) -> CausalOpsObservation:
        if extra_messages is None:
            extra_messages = []

        overview = {self._translate_to_agent(k): v.status for k, v in self._state.services.items()}

        detailed_metrics = {}
        for obs in self._state.agent.observations_made:
            if obs.startswith("metrics:"):
                svc = obs.split(":")[1]
                if svc in self._state.services:
                    raw = self._state.services[svc]
                    if raw.cpu_percent >= 99.0:
                        noisy_metrics = ServiceMetrics(
                            cpu_percent=100.0, memory_percent=raw.memory_percent,
                            latency_ms=0.0, error_rate=1.0, request_rate=0.0,
                            status="unreachable_telemetry"
                        )
                    else:
                        noisy_metrics = ServiceMetrics(
                            cpu_percent=self._add_noise(raw.cpu_percent, 2.0),
                            memory_percent=self._add_noise(raw.memory_percent, 1.0),
                            latency_ms=self._add_noise(raw.latency_ms, raw.latency_ms * 0.05),
                            error_rate=min(1.0, self._add_noise(raw.error_rate, 0.01)),
                            request_rate=self._add_noise(raw.request_rate, raw.request_rate * 0.02),
                            status=raw.status
                        )
                    detailed_metrics[self._translate_to_agent(svc)] = noisy_metrics

        tot_req = sum(s.request_rate for s in self._state.services.values())
        weighted_err = sum(s.error_rate * s.request_rate for s in self._state.services.values()) / max(1, tot_req)
        avg_lat = sum(s.latency_ms for s in self._state.services.values()) / max(1, len(self._state.services))
        healthy = sum(1 for s in self._state.services.values() if s.status == "healthy")
        degraded = sum(1 for s in self._state.services.values() if s.status == "degraded")
        critical = sum(1 for s in self._state.services.values() if s.status == "critical")

        agg = AggregateMetrics(
            total_request_rate=self._add_noise(tot_req, tot_req * 0.01),
            weighted_error_rate=min(1.0, self._add_noise(weighted_err, 0.005)),
            avg_latency_ms=self._add_noise(avg_lat, avg_lat * 0.02),
            services_healthy=healthy,
            services_degraded=degraded,
            services_critical=critical
        )

        logs = []
        for obs in self._state.agent.observations_made:
            if obs.startswith("logs:"):
                svc = obs.split(":")[1]
                raw_logs = self._task.logs_for_service(svc, self._state.step_number, self._state.services)
                logs.extend([self._translate_to_agent(l) for l in raw_logs])
            elif obs.startswith("traces:"):
                svc = obs.split(":")[1]
                if hasattr(self._task, 'traces_for_service'):
                    raw_traces = self._task.traces_for_service(svc, self._state.step_number, self._state.services)
                    logs.extend([self._translate_to_agent(t) for t in raw_traces])
            elif obs.startswith("config:"):
                svc = obs.split(":")[1]
                if hasattr(self._task, 'config_for_service'):
                    raw_config = self._task.config_for_service(svc, self._state.step_number, self._state.services)
                    logs.extend([self._translate_to_agent(c) for c in raw_config])

        translated_alerts = [self._translate_to_agent(a) for a in self._task.alerts_at_step(self._state.step_number)]

        all_msgs = self._task.stakeholder_messages_at_step(self._state.step_number) + extra_messages
        translated_msgs = []
        for m in all_msgs:
            translated_msgs.append(StakeholderMessage(
                sender=m.sender,
                message=self._translate_to_agent(m.message),
                requires_response=m.requires_response,
            ))

        translated_prompt = None
        if counterfactual_prompt:
            translated_prompt = CounterfactualPrompt(
                message=self._translate_to_agent(counterfactual_prompt.message),
                requires_prediction=counterfactual_prompt.requires_prediction
            )

        # Clamp reward to strictly (0, 1) exclusive when present
        clamped_reward = reward
        if clamped_reward is not None:
            clamped_reward = round(max(0.001, min(0.999, float(clamped_reward))), 3)

        return CausalOpsObservation(
            done=done,
            reward=clamped_reward,
            task_id=self._task_id,
            step_number=self._state.step_number,
            time_elapsed_s=self._state.time_elapsed_s,
            time_budget_remaining_s=self._state.time_budget_s - self._state.time_elapsed_s,
            services_overview=overview,
            detailed_metrics=detailed_metrics,
            aggregate_metrics=agg,
            alerts=translated_alerts,
            logs=logs,
            stakeholder_messages=translated_msgs,
            counterfactual_prompt=translated_prompt,
            task_description=self._translate_to_agent(self._task.description),
            available_actions=[e.value for e in ActionType]
        )

    def step(self, action: CausalOpsAction, **kwargs) -> CausalOpsObservation:
        if self._state.done:
            raise RuntimeError("Episode is done. Please reset().")

        assert self._dag is not None
        assert self._traps is not None

        self._state.step_number += 1
        self._state.step_count = self._state.step_number
        self._state.time_elapsed_s += self._task.time_step_s

        if self._state.time_elapsed_s >= self._state.time_budget_s or self._state.step_number >= self._task.max_steps:
            self._state.done = True

        action_type = action.type.value
        action_target = self._translate_from_agent(action.target)
        detail = self._translate_from_agent(action.detail)

        responded_this_step = False
        prompt: Optional[CounterfactualPrompt] = None

        if action.type == ActionType.OBSERVE:
            is_new_obs = action_target not in self._state.agent.observations_made
            if is_new_obs:
                self._state.agent.observations_made.append(action_target)

            obs_type = action_target.split(":")[0] if ":" in action_target else "metrics"
            cost = OBSERVE_COSTS.get(obs_type, 1)
            self._state.agent.total_observation_cost += cost

            if is_new_obs:
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
            self._state.time_elapsed_s += cost * 5.0

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
                pass

        elif action.type == ActionType.COMMUNICATE:
            self._state.agent.communications_sent.append(detail)
            responded_this_step = True
            if self._pending_responses > 0:
                self._pending_responses -= 1

        base_msgs = self._task.stakeholder_messages_at_step(self._state.step_number)
        extra_messages = []

        if self._state.agent.phantom_investigations > 0 and self._state.step_number % 10 == 0:
            extra_msg = StakeholderMessage(
                sender="vp_engineering",
                message="I see you looking at those metrics. Are you sure that's the root cause? The CEO is asking for updates. Fix the obvious spikes!",
                requires_response=True,
            )
            extra_messages.append(extra_msg)
            base_msgs.append(extra_msg)

        self._pending_responses += sum(1 for m in base_msgs if m.requires_response)

        self._state.services = self._task.evolve(self._state.services, self._state.step_number, self._state.agent.remediations_applied)

        all_healthy = all(s.status == "healthy" for s in self._state.services.values())
        if all_healthy and self._state.agent.remediations_applied:
            self._state.remediation_successful = True
            self._state.done = True

        reward_obj = compute_step_reward(
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
        self._state.total_reward += reward_obj.total

        # Determine the reward to return
        step_reward = reward_obj.total

        if self._state.done:
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
            # When done, reward = final grader score (what the platform uses)
            step_reward = final_scores["total"]

        obs = self._build_observation(
            counterfactual_prompt=prompt,
            extra_messages=extra_messages,
            reward=step_reward,
            done=self._state.done,
        )
        return obs

    @property
    def state(self) -> CausalOpsState:
        return self._state
