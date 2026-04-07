"""Abstract Task representation"""
from abc import ABC, abstractmethod
from typing import Dict, List
from causality import CausalDAG, ScenarioTraps
from models import ServiceMetrics, StakeholderMessage

class BaseTask(ABC):
    """Abstract base class for all CausalOps tasks."""
    
    task_id: str
    description: str
    time_budget_s: float
    max_steps: int = 30
    time_step_s: float = 10.0

    @abstractmethod
    def build_dag(self) -> CausalDAG:
        """Construct the ground-truth causal DAG."""
        pass

    @abstractmethod
    def build_traps(self) -> ScenarioTraps:
        """Construct the phantom scenario traps."""
        pass

    @abstractmethod
    def initial_services(self) -> Dict[str, ServiceMetrics]:
        """Return the initial metrics for all services."""
        pass

    @abstractmethod
    def evolve(
        self, 
        services: Dict[str, ServiceMetrics], 
        step: int, 
        remediations: List[str]
    ) -> Dict[str, ServiceMetrics]:
        """Evolve the service metrics by one time step."""
        pass

    @abstractmethod
    def stakeholder_messages_at_step(self, step: int) -> List[StakeholderMessage]:
        """Return any stakeholder messages injected at this step."""
        pass

    @abstractmethod
    def alerts_at_step(self, step: int) -> List[str]:
        """Return any monitoring alerts firing at this step."""
        pass

    @abstractmethod
    def logs_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Generate recent logs for a specific service."""
        pass

    @abstractmethod
    def expected_fix_effects(self) -> Dict[str, str]:
        """
        Return the expected counterfactual effects if the correct remediation is applied.
        Format: {"<metric>:<service>": "expected_change"}
        """
        pass

    def traces_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Generate OpenTelemetry-style trace spans for a specific service."""
        return []

    def config_for_service(self, svc: str, step: int, services: Dict[str, ServiceMetrics]) -> List[str]:
        """Generate recent configuration deployment logs."""
        return ["No recent deployments."]
