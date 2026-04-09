"""CausalOps OpenEnv client for interacting with the environment via HTTP or Docker."""
import os
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core import EnvClient
from models import CausalOpsAction, CausalOpsObservation, CausalOpsState

# Default Space URL — override via CAUSAL_OPS_URL env var
DEFAULT_URL = "http://localhost:7860"

ENV_URL = os.getenv("CAUSAL_OPS_URL", DEFAULT_URL)


class CausalOpsClient(EnvClient[CausalOpsAction, CausalOpsObservation, CausalOpsState]):
    """Thin wrapper around EnvClient pre-configured for CausalOps."""

    def __init__(self, base_url: str = ENV_URL):
        super().__init__(
            base_url=base_url,
        )

    def _step_payload(self, action: CausalOpsAction) -> Dict[str, Any]:
        return {
            "type": action.type.value,
            "target": action.target,
            "detail": action.detail,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CausalOpsObservation]:
        obs_data = payload.get("observation", payload)

        observation = CausalOpsObservation(**obs_data)

        # Ensure reward is fetched from payload if available
        reward = payload.get("reward", obs_data.get("reward", 0.0))
        done = payload.get("done", obs_data.get("done", False))
        
        # Make sure stepresult gets proper values
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CausalOpsState:
        return CausalOpsState(**payload.get("state", payload))

async def make_client(image_name: str | None = None, base_url: str | None = None) -> EnvClient:
    """Create a CausalOps client from a Docker image or URL.

    Args:
        image_name: Docker image name (uses from_docker_image).
        base_url: Direct URL to a running environment server.

    Returns:
        EnvClient configured for CausalOps.
    """
    if image_name:
        return await EnvClient.from_docker_image(
            image_name,
            action_cls=CausalOpsAction,
            observation_cls=CausalOpsObservation,
        )
    url = base_url or ENV_URL
    return CausalOpsClient(base_url=url)
