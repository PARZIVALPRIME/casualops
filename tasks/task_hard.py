"""Level 3: Regime change + Loops + Phantoms (The Shape-Shifter)"""
import random
from typing import Dict, List
from causality import CausalDAG, Edge, ScenarioTraps, Phantom
from models import ServiceMetrics, StakeholderMessage
from .base import BaseTask

class HardTask(BaseTask):
    task_id = "hard_shape_shifter"
    description = (
        "The Shape-Shifter: A complex system is experiencing massive performance degradation. "
        "The underlying cause may change over time, and feedback loops are present. "
        "Beware of multiple red herrings."
    )
    time_budget_s = 500.0
    max_steps = 50
    time_step_s = 10.0

    def build_dag(self) -> CausalDAG:
        nodes = ["frontend", "search-api", "inventory-db", "cache-node", "worker-svc"]
        # Phase 1 critical path: inventory-db -> search-api -> frontend
        # Cache node is part of a feedback loop (search-api -> cache-node -> search-api)
        edges = [
            Edge(source="inventory-db", target="search-api"),
            Edge(source="search-api", target="frontend"),
            Edge(source="search-api", target="cache-node"),
            Edge(source="cache-node", target="search-api"), # Feedback loop
        ]
        return CausalDAG(nodes=nodes, edges=edges, critical_path=[
            Edge(source="inventory-db", target="search-api"),
            Edge(source="search-api", target="frontend")
        ])

    def build_traps(self) -> ScenarioTraps:
        # 2 Phantoms: CDN latency and Cron job spikes
        return ScenarioTraps(phantoms=[
            Phantom(phantom_node="cdn", correlated_node="inventory-db"),
            Phantom(phantom_node="worker-svc", correlated_node="search-api")
        ])

    def initial_services(self) -> Dict[str, ServiceMetrics]:
        return {
            "frontend": ServiceMetrics(
                cpu_percent=30.0, memory_percent=40.0, latency_ms=80.0, 
                error_rate=0.0, request_rate=10000.0, status="healthy"
            ),
            "search-api": ServiceMetrics(
                cpu_percent=40.0, memory_percent=45.0, latency_ms=120.0, 
                error_rate=0.0, request_rate=5000.0, status="healthy"
            ),
            "inventory-db": ServiceMetrics(
                cpu_percent=50.0, memory_percent=60.0, latency_ms=20.0, 
                error_rate=0.0, request_rate=8000.0, status="healthy"
            ),
            "cache-node": ServiceMetrics(
                cpu_percent=20.0, memory_percent=80.0, latency_ms=5.0, 
                error_rate=0.0, request_rate=15000.0, status="healthy"
            ),
            "worker-svc": ServiceMetrics(  # Represents cron_job_spike
                cpu_percent=10.0, memory_percent=20.0, latency_ms=5.0, 
                error_rate=0.0, request_rate=100.0, status="healthy"
            ),
            "cdn": ServiceMetrics(       # Represents cdn_latency
                cpu_percent=15.0, memory_percent=10.0, latency_ms=10.0, 
                error_rate=0.0, request_rate=12000.0, status="healthy"
            )
        }

    def evolve(
        self, services: Dict[str, ServiceMetrics], step: int, remediations: List[str]
    ) -> Dict[str, ServiceMetrics]:
        front = services["frontend"]
        search = services["search-api"]
        db = services["inventory-db"]
        cache = services["cache-node"]
        worker = services["worker-svc"]
        cdn = services["cdn"]

        # Phase 1: DB degrades
        # Phase 2 (step >= 13): Cache thrashes + DB stabilizes
        
        fixed_db = "scale:inventory-db" in remediations
        fixed_cache = "restart:cache-node" in remediations or "config:cache-node" in remediations
        
        if step < 13:
            # PHASE 1: DB degradation
            if fixed_db:
                db.latency_ms = max(20.0, db.latency_ms - 100.0)
                db.status = "healthy"
            else:
                db.latency_ms = min(600.0, db.latency_ms + 100.0)
                db.status = "critical" if db.latency_ms > 300 else "degraded"
                
                # Phantom 1: CDN correlates with DB
                cdn.latency_ms = min(200.0, cdn.latency_ms + 30.0)
                cdn.status = "degraded"

            # Propagation
            search.latency_ms = db.latency_ms + 100.0
            search.status = "critical" if search.latency_ms > 400 else "degraded"
            
            front.latency_ms = search.latency_ms + 50.0
            front.error_rate = min(0.15, front.error_rate + 0.02)
            front.status = search.status
            
        else:
            # PHASE 2: Regime Change (Cache Thrashing Feedback Loop)
            # DB recovers naturally (or due to fix), but cache dies
            if not fixed_db:
                db.latency_ms = max(20.0, db.latency_ms - 150.0)
                db.status = "healthy" if db.latency_ms < 100 else "degraded"
                cdn.latency_ms = max(10.0, cdn.latency_ms - 50.0)
                cdn.status = "healthy"
                
            if fixed_cache:
                cache.cpu_percent = max(20.0, cache.cpu_percent - 50.0)
                cache.error_rate = 0.0
                cache.status = "healthy"
                
                search.latency_ms = max(120.0, search.latency_ms - 200.0)
                search.status = "healthy"
                
                front.latency_ms = max(80.0, front.latency_ms - 200.0)
                front.error_rate = max(0.0, front.error_rate - 0.05)
                front.status = "healthy" if front.error_rate < 0.02 else "degraded"
                
                worker.cpu_percent = max(10.0, worker.cpu_percent - 30.0)
            else:
                # Feedback loop: cache misses -> search load -> cache evictions -> cache misses
                cache.cpu_percent = min(100.0, cache.cpu_percent + 20.0)
                cache.error_rate = min(0.5, cache.error_rate + 0.1)
                cache.status = "critical"
                
                # Phantom 2: Cron job spikes correlate with Search API degradation
                worker.cpu_percent = min(90.0, worker.cpu_percent + 25.0)
                
                search.latency_ms = cache.cpu_percent * 8.0
                search.status = "critical"
                
                front.latency_ms = search.latency_ms + cache.latency_ms
                front.error_rate = cache.error_rate * 0.5
                front.status = "critical"
                
        return services

    def stakeholder_messages_at_step(self, step: int) -> List[StakeholderMessage]:
        msgs = []
        if step == 15:
            msgs.append(StakeholderMessage(
                sender="product_manager",
                message="The site is completely unusable! The DB team said they fixed it, why is it still broken?",
                requires_response=True
            ))
        if step == 18:
            msgs.append(StakeholderMessage(
                sender="other_engineer",
                message="Look at the cron jobs (worker-svc)! CPU is at 90%! We should kill the cron jobs right now.",
                requires_response=False,
                is_adversarial=True
            ))
        return msgs

    def alerts_at_step(self, step: int) -> List[str]:
        if step >= 5 and step < 13:
            return ["DbLatencyHigh (inventory-db)", "SearchApiSlow (search-api)"]
        if step >= 13:
            return ["CacheEvictionSpike (cache-node)", "SearchApiSlow (search-api)", "FrontendErrors (frontend)"]
        return []

    def logs_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        if svc == "inventory-db" and step < 13:
            return ["Lock wait timeout exceeded"] * 2
        if svc == "cache-node" and step >= 13:
            return ["OOM command not allowed when used memory > 'maxmemory'", "Evicting keys"] * 2
        if svc == "search-api" and step >= 13:
            return ["Timeout waiting for cache response", "Cache miss, reading from DB"] * 2
        if svc == "worker-svc" and step >= 15:
            return ["Running batch job: cleanup_orphans"] * 3
        return []

    def traces_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Industry Grade: Distributed traces for Regime Change scenarios."""
        if step < 5:
            return []
        trace_id = f"tr-h3{step}"
        if svc == "frontend":
            return [f'[{trace_id}] span_id=501 name="fe:render" kind=SERVER duration_ms=1200 status=200',
                    f'[{trace_id}] span_id=502 parent_id=501 name="search:call" kind=CLIENT duration_ms=1150 status=warn']
        if svc == "search-api":
            if step < 13:
                return [f'[{trace_id}] span_id=601 parent_id=502 name="search:exec" kind=SERVER duration_ms=1150 status=200',
                        f'[{trace_id}] span_id=602 parent_id=601 name="db:lookup" kind=CLIENT duration_ms=1100 status=timeout']
            else:
                return [f'[{trace_id}] span_id=701 parent_id=502 name="search:exec" kind=SERVER duration_ms=2500 status=500',
                        f'[{trace_id}] span_id=702 parent_id=701 name="cache:get" kind=CLIENT duration_ms=2400 status=error']
        if svc == "inventory-db":
            return [f'[{trace_id}] span_id=801 parent_id=602 name="db:query" kind=SERVER duration_ms=1100 query="UPDATE inv" status=LOCK_WAIT']
        if svc == "cache-node":
            return [f'[{trace_id}] span_id=901 parent_id=702 name="cache:op" kind=SERVER duration_ms=2400 status=EVICTION_LOOP']
        return []

    def config_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Industry Grade: Phantom Configs for CDN and Workers."""
        if svc == "cdn":
            return [f"2026-04-08 09:15:00 - DEPLOY - env=prod - actor=frontend-bot - commit=f1e2d - msg='Invalidate CDN cache (PHANTOM)'"]
        if svc == "worker-svc":
            return [f"2026-04-08 10:10:00 - DEPLOY - env=prod - actor=devops - commit=5a4b3 - msg='Scale worker replicas (PHANTOM)'"]
        return ["No deployments in last 24h."]

    def expected_fix_effects(self) -> Dict[str, str]:
        return {"error_rate:cache-node": "-0.5,10"}
