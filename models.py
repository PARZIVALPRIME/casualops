"""Pydantic models for CausalOps OpenEnv environment."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Action Types ──────────────────────────────────────────────────────
class ActionType(str, Enum):
    OBSERVE = "observe"            # Request metrics/logs for a service
    HYPOTHESIZE = "hypothesize"    # Claim a causal edge (cause->effect)
    REMEDIATE = "remediate"        # Apply a fix to a service
    COMMUNICATE = "communicate"    # Respond to a stakeholder
    PREDICT = "predict"            # Counterfactual prediction


class Action(BaseModel):
    """Agent action submitted to the environment each step.

    Extends openenv.core.Action interface with metadata support.
    """
    model_config = ConfigDict(extra="allow")

    type: ActionType
    target: str = Field(
        ...,
        description=(
            "observe  → 'metrics:<svc>' | 'logs:<svc>' | 'traces:<svc>' | 'config:<svc>'\n"
            "hypothesize → '<cause_node>-><effect_node>'\n"
            "remediate → 'restart:<svc>' | 'scale:<svc>' | 'config:<svc>'\n"
            "communicate → '<stakeholder_id>'\n"
            "predict → '<metric>:<svc>'"
        ),
    )
    detail: str = Field(
        "",
        description=(
            "Extra payload.\n"
            "  communicate → response text\n"
            "  predict     → '<expected_delta>,<timeframe_s>'\n"
            "  hypothesize → optional confidence (0-1)"
        ),
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Observation costs (information acquisition) ─────────────────────
OBSERVE_COSTS: Dict[str, int] = {
    "metrics": 1,
    "logs": 2,
    "traces": 3,
    "config": 2,
    "diagnostic": 4,
}

REMEDIATE_COSTS: Dict[str, int] = {
    "restart": 5,
    "scale": 4,
    "config": 3,
    "rollback": 4,
}


# ── Service-level metrics ────────────────────────────────────────────
class ServiceMetrics(BaseModel):
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    request_rate: float = 100.0
    disk_io_percent: float = 0.0
    status: str = "healthy"  # healthy | degraded | critical


# ── Stakeholder messages ─────────────────────────────────────────────
class StakeholderMessage(BaseModel):
    sender: str           # product_manager | vp_engineering | other_engineer
    message: str
    requires_response: bool = False


# ── Observation (returned to agent) ──────────────────────────────────
class AggregateMetrics(BaseModel):
    """System-wide aggregate metrics (may be misleading — Simpson's Paradox)."""
    total_request_rate: float = 0.0
    weighted_error_rate: float = 0.0     # traffic-weighted aggregate
    avg_latency_ms: float = 0.0
    services_healthy: int = 0
    services_degraded: int = 0
    services_critical: int = 0


class CounterfactualPrompt(BaseModel):
    """Injected after a remediate action — demands a falsifiable prediction."""
    message: str
    requires_prediction: bool = True


class Observation(BaseModel):
    """OpenEnv-compatible observation with done/reward at top level."""
    model_config = ConfigDict(extra="allow")

    # OpenEnv required fields
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # CausalOps fields
    task_id: str = ""
    step_number: int = 0
    time_elapsed_s: float = 0.0
    time_budget_remaining_s: float = 0.0
    services_overview: Dict[str, str] = {}        # svc → status string
    detailed_metrics: Dict[str, ServiceMetrics] = {}  # only observed svcs
    aggregate_metrics: Optional[AggregateMetrics] = None  # system-wide view
    alerts: List[str] = []
    logs: List[str] = []
    stakeholder_messages: List[StakeholderMessage] = []
    counterfactual_prompt: Optional[CounterfactualPrompt] = None
    task_description: str = ""
    available_actions: List[str] = []


# ── Reward (per-step, dense) ─────────────────────────────────────────
class RewardComponents(BaseModel):
    hypothesis_quality: float = Field(default=0.0, ge=0.0, le=0.3)
    information_efficiency: float = Field(default=0.0, ge=0.0, le=0.2)
    phantom_resistance: float = Field(default=0.0, ge=0.0, le=0.2)
    communication_quality: float = Field(default=0.0, ge=0.0, le=0.15)
    time_pressure_management: float = Field(default=0.0, ge=0.0, le=0.15)

    @property
    def total(self) -> float:
        return (
            self.hypothesis_quality
            + self.information_efficiency
            + self.phantom_resistance
            + self.communication_quality
            + self.time_pressure_management
        )


class Reward(BaseModel):
    components: RewardComponents = Field(default_factory=lambda: RewardComponents(
        hypothesis_quality=0.0,
        information_efficiency=0.0,
        phantom_resistance=0.0,
        communication_quality=0.0,
        time_pressure_management=0.0
    ))
    total: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""


# ── Agent-side accumulated state ─────────────────────────────────────
class CausalClaim(BaseModel):
    cause: str
    effect: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class CounterfactualPrediction(BaseModel):
    metric_name: str
    service: str
    expected_delta: float
    timeframe_s: float
    step_made: int = 0


class AgentState(BaseModel):
    hypotheses: List[CausalClaim] = []
    predictions: List[CounterfactualPrediction] = []
    observations_made: List[str] = []
    remediations_applied: List[str] = []
    communications_sent: List[str] = []
    total_observation_cost: int = 0
    useful_observations: int = 0
    phantom_investigations: int = 0


# ── Full environment state ───────────────────────────────────────────
class State(BaseModel):
    model_config = ConfigDict(extra="allow")

    # OpenEnv required fields
    episode_id: Optional[str] = None
    step_count: int = 0

    # CausalOps fields
    task_id: str = ""
    step_number: int = 0
    done: bool = False
    services: Dict[str, ServiceMetrics] = {}
    agent: AgentState = Field(default_factory=AgentState)
    time_elapsed_s: float = 0.0
    time_budget_s: float = 300.0
    current_phase: int = 1
    total_reward: float = 0.0
    remediation_successful: bool = False


# ── Step result bundle (internal use) ───────────────────────────────
class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, object] = {}
