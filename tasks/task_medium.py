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
        """Industry Grade: Detailed OpenTelemetry-style traces."""
        if step < 4:
            return []
        trace_id = f"tr-9e{step}"
        if svc == "api-gateway":
            return [
                f'[{trace_id}] span_id=01 name="api:request" kind=SERVER duration_ms=1500 status=503',
                f'[{trace_id}] span_id=02 parent_id=01 name="auth:call" kind=CLIENT duration_ms=1490 status=error'
            ]
        if svc == "auth-service":
            return [
                f'[{trace_id}] span_id=03 parent_id=02 name="auth:process" kind=SERVER duration_ms=1490 status=500',
                f'[{trace_id}] span_id=04 parent_id=03 name="db:query" kind=CLIENT duration_ms=1450 status=timeout'
            ]
        if svc == "user-db":
            return [
                f'[{trace_id}] span_id=05 parent_id=04 name="db:execute" kind=SERVER duration_ms=1450 query="SELECT * FROM users" status=ERROR_OOM'
            ]
        if svc == "payment-gateway":
            p_trace = f"tr-p{step}"
            return [
                f'[{p_trace}] span_id=10 name="pay:init" kind=SERVER duration_ms=2000 status=503',
                f'[{p_trace}] span_id=11 parent_id=10 name="auth:validate" kind=CLIENT duration_ms=1990 status=error'
            ]
        return []

    def config_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Industry Grade: GitOps config history with 'Phantom Configs'."""
        if svc == "dns-resolver":
            return [
                f"2026-04-08 10:{step:02d}:05 - DEPLOY - env=prod - actor=gitops - commit=9a8b7c - msg='Update CoreDNS config (PHANTOM)'",
                "2026-04-07 14:22:10 - DEPLOY - env=prod - actor=gitops - commit=1a2b3c - msg='Initial baseline'"
            ]
        if svc == "user-db":
            return ["2026-04-01 09:00:00 - DEPLOY - env=prod - actor=dba - commit=f0f0f0 - msg='Monthly maintenance'"]
        
        return ["No recent deployments in the last 24h."]

    def expected_fix_effects(self) -> Dict[str, str]:
        return {"latency_ms:user-db": "-500.0,20"}
