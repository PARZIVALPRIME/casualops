"""Level 2: Branching DAG + 1 Phantom (The Web of Lies)"""
import random
from typing import Dict, List
from causality import CausalDAG, Edge, ScenarioTraps, Phantom
from models import ServiceMetrics, StakeholderMessage
from .base import BaseTask

class MediumTask(BaseTask):
    task_id = "medium_web_of_lies"
    description = (
        "The Web of Lies: Multiple services are failing. "
        "DNS latency is spiking alongside API errors. "
        "Find the true root cause and ignore the red herrings."
    )
    time_budget_s = 400.0
    max_steps = 40
    time_step_s = 10.0

    def build_dag(self) -> CausalDAG:
        nodes = ["api-gateway", "auth-service", "user-db", "payment-gateway"]
        # Root cause: user-db is overwhelmed -> auth-service fails -> api-gateway fails
        # payment-gateway also fails because it depends on auth-service
        edges = [
            Edge(source="user-db", target="auth-service"),
            Edge(source="auth-service", target="api-gateway"),
            Edge(source="auth-service", target="payment-gateway")
        ]
        # Critical path is user-db -> auth-service -> api-gateway
        return CausalDAG(nodes=nodes, edges=edges, critical_path=[
            Edge(source="user-db", target="auth-service"),
            Edge(source="auth-service", target="api-gateway")
        ])

    def build_traps(self) -> ScenarioTraps:
        # 1 Phantom: DNS latency spikes because of the same underlying load that hit the user-db,
        # but DNS is NOT causing the service failures. 
        # Here we correlate it to 'user-db' to make it temporal.
        return ScenarioTraps(phantoms=[
            Phantom(phantom_node="dns-resolver", correlated_node="user-db")
        ])

    def initial_services(self) -> Dict[str, ServiceMetrics]:
        return {
            "api-gateway": ServiceMetrics(
                cpu_percent=45.0, memory_percent=35.0, latency_ms=40.0, 
                error_rate=0.0, request_rate=5000.0, status="healthy"
            ),
            "auth-service": ServiceMetrics(
                cpu_percent=55.0, memory_percent=60.0, latency_ms=60.0, 
                error_rate=0.0, request_rate=5000.0, status="healthy"
            ),
            "user-db": ServiceMetrics(
                cpu_percent=40.0, memory_percent=50.0, latency_ms=15.0, 
                error_rate=0.0, request_rate=10000.0, status="healthy"
            ),
            "payment-gateway": ServiceMetrics(
                cpu_percent=30.0, memory_percent=30.0, latency_ms=150.0, 
                error_rate=0.0, request_rate=500.0, status="healthy"
            ),
            "dns-resolver": ServiceMetrics(
                cpu_percent=20.0, memory_percent=20.0, latency_ms=5.0, 
                error_rate=0.0, request_rate=20000.0, status="healthy"
            ),
        }

    def evolve(
        self, services: Dict[str, ServiceMetrics], step: int, remediations: List[str]
    ) -> Dict[str, ServiceMetrics]:
        db = services["user-db"]
        auth = services["auth-service"]
        api = services["api-gateway"]
        pay = services["payment-gateway"]
        dns = services["dns-resolver"]

        fixed = "scale:user-db" in remediations or "config:user-db" in remediations

        if fixed:
            # Gradual recovery
            db.latency_ms = max(15.0, db.latency_ms - 200.0)
            db.cpu_percent = max(40.0, db.cpu_percent - 20.0)
            db.status = "healthy" if db.latency_ms < 50 else "degraded"
            
            auth.latency_ms = max(60.0, auth.latency_ms - 300.0)
            auth.error_rate = max(0.0, auth.error_rate - 0.1)
            auth.status = "healthy" if auth.error_rate < 0.05 else "degraded"
            
            api.error_rate = auth.error_rate
            api.status = "healthy" if api.error_rate < 0.05 else "degraded"
            
            pay.error_rate = auth.error_rate
            pay.status = "healthy" if pay.error_rate < 0.05 else "degraded"

            dns.latency_ms = max(5.0, dns.latency_ms - 100.0)
            dns.status = "healthy"
            
        elif step >= 4:
            # Degradation manifests
            # user-db degrades first
            db.cpu_percent = min(95.0, db.cpu_percent + 15.0)
            db.latency_ms = min(800.0, db.latency_ms + 150.0)
            db.status = "critical" if db.latency_ms > 400 else "degraded"
            
            # DNS degrades simultaneously (Phantom)
            dns.latency_ms = min(300.0, dns.latency_ms + 60.0)
            dns.status = "degraded"
            
            if db.latency_ms > 200:
                auth.latency_ms = db.latency_ms + 100.0
                auth.error_rate = min(0.4, auth.error_rate + 0.05)
                auth.status = "critical" if auth.error_rate > 0.15 else "degraded"
            
            if auth.error_rate > 0.05:
                api.error_rate = auth.error_rate
                api.status = auth.status
                
                pay.error_rate = auth.error_rate
                pay.status = auth.status

        return services

    def stakeholder_messages_at_step(self, step: int) -> List[StakeholderMessage]:
        msgs = []
        if step == 6:
            msgs.append(StakeholderMessage(
                sender="vp_engineering",
                message="Multiple services are down. Update me in 5 minutes.",
                requires_response=True
            ))
        if step == 8:
            msgs.append(StakeholderMessage(
                sender="other_engineer",
                message="Hey, I'm looking at the dashboards. DNS latency is through the roof! Should we restart the dns-resolver?",
                requires_response=False,
                is_adversarial=True
            ))
        return msgs

    def alerts_at_step(self, step: int) -> List[str]:
        if step >= 5:
            return [
                "ApiGateway5xxSpike (api-gateway)", 
                "AuthFailures (auth-service)", 
                "DnsResolutionSlow (dns-resolver)"
            ]
        return []

    def logs_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        if step < 4:
            return []
        if svc == "api-gateway":
            return ['503 Service Unavailable: auth-service disconnected'] * 2
        if svc == "auth-service":
            return ['Timeout connecting to user-db pool', 'Failed to authenticate user token'] * 2
        if svc == "user-db":
            return ['FATAL: out of memory for query', 'Index scan extremely slow'] * 2
        if svc == "dns-resolver":
            return ['WARN: resolver queue full', 'Query taking > 200ms'] * 2
        return []

    def traces_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        if step < 4:
            return []
        if svc == "api-gateway":
            return ['{"trace_id": "c9d8", "span": "api-gateway", "duration_ms": 1500, "downstream": "auth-service", "status": "ERROR"}']
        if svc == "auth-service":
            return ['{"trace_id": "c9d8", "span": "auth-service", "duration_ms": 1490, "downstream": "user-db", "status": "ERROR"}']
        if svc == "user-db":
            return ['{"trace_id": "c9d8", "span": "user-db", "duration_ms": 1450, "query": "SELECT user_profiles", "status": "OOM_KILLED"}']
        if svc == "payment-gateway":
            return ['{"trace_id": "x7y6", "span": "payment-gateway", "duration_ms": 2000, "downstream": "auth-service", "status": "ERROR"}']
        return []

    def expected_fix_effects(self) -> Dict[str, str]:
        return {"latency_ms:user-db": "-500.0,20"}
