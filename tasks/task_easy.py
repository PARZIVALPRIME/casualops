"""Level 1: Linear chain (The Smoking Gun)"""
import random
from typing import Dict, List
from causality import CausalDAG, Edge, ScenarioTraps
from models import ServiceMetrics, StakeholderMessage
from .base import BaseTask

class EasyTask(BaseTask):
    task_id = "easy_smoking_gun"
    description = (
        "The Smoking Gun: LoadBalancer -> AppServer -> Database. "
        "The system is experiencing cascading timeouts. Diagnose the root cause."
    )
    time_budget_s = 300.0
    max_steps = 30
    time_step_s = 10.0

    def build_dag(self) -> CausalDAG:
        nodes = ["load-balancer", "app-server", "database"]
        # Linear chain: db is slow -> app times out -> lb shows errors
        edges = [
            Edge(source="database", target="app-server"),
            Edge(source="app-server", target="load-balancer"),
        ]
        return CausalDAG(nodes=nodes, edges=edges, critical_path=edges)

    def build_traps(self) -> ScenarioTraps:
        # Easy task has no phantoms
        return ScenarioTraps(phantoms=[])

    def initial_services(self) -> Dict[str, ServiceMetrics]:
        return {
            "load-balancer": ServiceMetrics(
                cpu_percent=40.0, memory_percent=30.0, latency_ms=50.0, 
                error_rate=0.0, request_rate=1000.0, status="healthy"
            ),
            "app-server": ServiceMetrics(
                cpu_percent=60.0, memory_percent=50.0, latency_ms=100.0, 
                error_rate=0.0, request_rate=1000.0, status="healthy"
            ),
            "database": ServiceMetrics(
                cpu_percent=30.0, memory_percent=60.0, latency_ms=10.0, 
                error_rate=0.0, request_rate=2000.0, status="healthy"
            ),
        }

    def evolve(
        self, services: Dict[str, ServiceMetrics], step: int, remediations: List[str]
    ) -> Dict[str, ServiceMetrics]:
        lb = services["load-balancer"]
        app = services["app-server"]
        db = services["database"]

        fixed = "restart:database" in remediations or "scale:database" in remediations or "config:database" in remediations

        if fixed:
            # Gradual recovery
            db.latency_ms = max(10.0, db.latency_ms - 100.0)
            db.cpu_percent = max(30.0, db.cpu_percent - 10.0)
            db.status = "healthy"
            
            app.latency_ms = max(100.0, app.latency_ms - 200.0)
            app.error_rate = max(0.0, app.error_rate - 0.05)
            app.status = "healthy" if app.error_rate < 0.02 else "degraded"
            
            lb.error_rate = app.error_rate
            lb.status = "healthy" if lb.error_rate < 0.02 else "degraded"
            
        elif step >= 3:
            # Degradation manifests
            db.cpu_percent = min(99.0, db.cpu_percent + 20.0)
            db.latency_ms = min(500.0, db.latency_ms + 100.0)
            db.status = "critical" if db.latency_ms > 300 else "degraded"
            
            app.latency_ms = db.latency_ms + 50.0
            app.error_rate = min(0.2, app.error_rate + 0.05)
            app.status = "critical" if app.error_rate > 0.1 else "degraded"
            
            lb.error_rate = app.error_rate
            lb.status = app.status

        return services

    def stakeholder_messages_at_step(self, step: int) -> List[StakeholderMessage]:
        if step == 5:
            return [StakeholderMessage(
                sender="product_manager",
                message="Users are reporting 500 errors on checkout. What's the ETA?",
                requires_response=True
            )]
        return []

    def alerts_at_step(self, step: int) -> List[str]:
        if step >= 4:
            return ["HighErrorRate (load-balancer)", "LatencySpike (app-server)"]
        return []

    def logs_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        if step < 3:
            return []
        if svc == "load-balancer":
            return ['nginx: 502 Bad Gateway - upstream timed out'] * 3
        if svc == "app-server":
            return ['Error: Connection timeout to database (3000ms)'] * 3
        if svc == "database":
            return ['WARN: slow query detected', 'ERROR: max connections reached'] * 2
        return []

    def traces_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        if step < 3:
            return []
        if svc == "load-balancer":
            return ['{"trace_id": "a1b2", "span": "load-balancer", "duration_ms": 3050, "status": "ERROR"}']
        if svc == "app-server":
            return ['{"trace_id": "a1b2", "span": "app-server", "duration_ms": 3000, "downstream": "database", "status": "ERROR"}']
        if svc == "database":
            return ['{"trace_id": "a1b2", "span": "database", "duration_ms": 3000, "query": "SELECT *", "status": "TIMEOUT"}']
        return []

    def expected_fix_effects(self) -> Dict[str, str]:
        # "If this fix works, what metric should change first, by how much, and in what timeframe?"
        return {"latency_ms:database": "-400.0,30"}
