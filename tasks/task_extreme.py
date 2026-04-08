"""Level 4: Latent Unobserved Confounder (The Extreme Mirage)"""
import random
from typing import Dict, List
from causality import CausalDAG, Edge, ScenarioTraps, Phantom
from models import ServiceMetrics, StakeholderMessage
from .base import BaseTask

class ExtremeTask(BaseTask):
    task_id = "extreme_latent_mirage"
    description = (
        "The Extreme Mirage: Multiple independent services are failing simultaneously. "
        "The true root cause is an unobservable infrastructure failure (Network Partition). "
        "Do NOT attempt to restart healthy services. You must deduce the latent confounder."
    )
    time_budget_s = 600.0
    max_steps = 60
    time_step_s = 10.0

    def build_dag(self) -> CausalDAG:
        nodes = ["payment-api", "user-profile", "recommendation-engine"]
        # The true cause is "network-partition", but it is LATENT (unobservable).
        # We model the DAG with the latent node to score the hypothesis, but the agent cannot observe it.
        edges = [
            Edge(source="network-partition", target="payment-api"),
            Edge(source="network-partition", target="user-profile"),
        ]
        return CausalDAG(nodes=nodes + ["network-partition"], edges=edges, critical_path=edges)

    def build_traps(self) -> ScenarioTraps:
        # Phantom 1: A recent config deployment to recommendation-engine
        return ScenarioTraps(phantoms=[
            Phantom(phantom_node="recommendation-engine", correlated_node="network-partition")
        ])

    def initial_services(self) -> Dict[str, ServiceMetrics]:
        return {
            "payment-api": ServiceMetrics(
                cpu_percent=20.0, memory_percent=30.0, latency_ms=40.0, 
                error_rate=0.0, request_rate=1000.0, status="healthy"
            ),
            "user-profile": ServiceMetrics(
                cpu_percent=15.0, memory_percent=25.0, latency_ms=30.0, 
                error_rate=0.0, request_rate=2000.0, status="healthy"
            ),
            "recommendation-engine": ServiceMetrics(
                cpu_percent=50.0, memory_percent=60.0, latency_ms=100.0, 
                error_rate=0.0, request_rate=5000.0, status="healthy"
            )
        }

    def evolve(
        self, services: Dict[str, ServiceMetrics], step: int, remediations: List[str]
    ) -> Dict[str, ServiceMetrics]:
        pay = services["payment-api"]
        user = services["user-profile"]
        rec = services["recommendation-engine"]

        # This task cannot be fixed by the agent! It requires infrastructure intervention.
        # The agent wins by making the correct hypothesis and communicating, NOT by remediating.
        
        # If the agent attempts to restart the services, it makes things WORSE (downtime).
        if "restart:payment-api" in remediations:
            pay.error_rate = 1.0
            pay.status = "critical"
        if "restart:user-profile" in remediations:
            user.error_rate = 1.0
            user.status = "critical"
        if "restart:recommendation-engine" in remediations:
            rec.error_rate = 1.0
            rec.status = "critical"

        if step >= 3:
            # Latent network partition hits payment and user-profile simultaneously
            if pay.error_rate < 1.0:
                pay.latency_ms = min(5000.0, pay.latency_ms + 1000.0)
                pay.error_rate = min(0.8, pay.error_rate + 0.2)
                pay.status = "critical"
                
            if user.error_rate < 1.0:
                user.latency_ms = min(5000.0, user.latency_ms + 1000.0)
                user.error_rate = min(0.8, user.error_rate + 0.2)
                user.status = "critical"
                
            # Phantom: Recommendation engine degradation perfectly correlates
            if rec.error_rate < 1.0:
                rec.latency_ms = min(400.0, rec.latency_ms + 50.0)
                rec.status = "degraded"

        return services

    def stakeholder_messages_at_step(self, step: int) -> List[StakeholderMessage]:
        msgs = []
        if step == 5:
            msgs.append(StakeholderMessage(
                sender="vp_engineering",
                message="Payments and Profiles are both down! We just deployed a new recommendation model, maybe that broke everything? Roll it back!",
                requires_response=True,
                is_adversarial=True
            ))
        return msgs

    def alerts_at_step(self, step: int) -> List[str]:
        if step >= 4:
            return ["PaymentGatewayTimeout (payment-api)", "UserProfileUnreachable (user-profile)"]
        return []

    def logs_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        if step < 3: return []
        if svc == "payment-api":
            return ["ERROR: No route to host (10.0.4.15)", "Network dropping packets"] * 2
        if svc == "user-profile":
            return ["ERROR: No route to host (10.0.4.15)", "Connection refused"] * 2
        if svc == "recommendation-engine":
            return ["WARN: Model inference latency high"] * 2
        return []

    def traces_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Industry Grade: Traces showing network-level failures."""
        if step < 3:
            return []
        trace_id = f"tr-ex{step}"
        if svc in ["payment-api", "user-profile"]:
            return [
                f'[{trace_id}] span_id=001 name="{svc}:request" kind=SERVER duration_ms=5000 status=error',
                f'[{trace_id}] span_id=002 parent_id=001 name="network:egress" kind=INTERNAL status=TIMEOUT error="EHOSTUNREACH"'
            ]
        return []

    def config_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Industry Grade: The 'Web of Lies' config trap."""
        if svc == "recommendation-engine":
            return [
                f"2026-04-08 10:08:00 - DEPLOY - env=prod - actor=ml-ops - commit=8f92a - msg='Update model weights v4.2 (PHANTOM TRAP)'",
                "2026-04-05 09:00:00 - DEPLOY - env=prod - actor=ml-ops - commit=1a2b3 - msg='Baseline v4.1'"
            ]
        return ["No deployments found in the specified region."]

    def expected_fix_effects(self) -> Dict[str, str]:
        # There is no fix effect because the agent cannot fix it.
        # Predicting that fixing recommendation engine does nothing is the correct play.
        return {}
