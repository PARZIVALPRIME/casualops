"""FastAPI layer for CausalOps OpenEnv environment.

Thin HTTP wrapper — all logic lives in env/environment.py.
Returns responses in OpenEnv-compatible format:
  POST /reset  → {observation: {...}, reward: null, done: false}
  POST /step   → {observation: {...}, reward: float, done: bool}
  GET  /state  → State
  GET  /tasks  → task list
  GET  /health → health check
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import CausalOpsEnvironment
from models import Action, Observation, State

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(
    title="CausalOps — Causal Inference Gym",
    description=(
        "The first OpenEnv environment that benchmarks AI agents' ability "
        "to perform real-time causal reasoning in adversarial production "
        "system scenarios. Where right answers for wrong reasons score zero."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single global environment instance (stateful per session)
_env = CausalOpsEnvironment()


# ── Request models ───────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = "easy_smoking_gun"
    seed: int | None = None
    episode_id: str | None = None


# ── OpenEnv-compatible response models ───────────────────────────
class OpenEnvResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class OpenEnvStepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


def _serialize_observation(obs: Observation) -> Dict[str, Any]:
    """Serialize observation, excluding done/reward/metadata (OpenEnv convention)."""
    return obs.model_dump(exclude={"done", "reward", "metadata"})


# ── Endpoints ────────────────────────────────────────────────────
@app.post("/reset", response_model=OpenEnvResetResponse)
def reset(req: ResetRequest | None = None) -> OpenEnvResetResponse:
    """Initialize a new episode for the given task."""
    if req is None:
        req = ResetRequest()
    try:
        obs = _env.reset(req.task_id, seed=req.seed)
        return OpenEnvResetResponse(
            observation=_serialize_observation(obs),
            reward=None,
            done=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=OpenEnvStepResponse)
def step(action: Action) -> OpenEnvStepResponse:
    """Execute one agent action and advance the environment."""
    try:
        result = _env.step(action)
        return OpenEnvStepResponse(
            observation=_serialize_observation(result.observation),
            reward=result.reward.total,
            done=result.done,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def state() -> State:
    """Return the full current environment state."""
    try:
        return _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks() -> dict:
    """List all available task IDs."""
    return {"tasks": _env.available_tasks}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CausalOps - The Causal Inference Gym</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 0; }
            .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
            h1 { color: #58a6ff; font-size: 2.5em; margin-bottom: 0.2em; }
            h2 { color: #79c0ff; margin-top: 1.5em; border-bottom: 1px solid #30363d; padding-bottom: 0.3em; }
            p { line-height: 1.6; color: #8b949e; }
            .highlight { color: #f0883e; font-weight: bold; }
            .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
            .mermaid { background-color: #0d1117; padding: 15px; border-radius: 6px; }
            .api-link { display: inline-block; padding: 10px 20px; background-color: #238636; color: white; text-decoration: none; border-radius: 6px; font-weight: bold; margin-top: 20px; }
            .api-link:hover { background-color: #2ea043; }
        </style>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({ startOnLoad: true, theme: 'dark' });
        </script>
    </head>
    <body>
        <div class="container">
            <h1>CausalOps</h1>
            <p><strong>The Causal Inference Gym for Production Systems</strong></p>
            <p>Welcome to the first OpenEnv benchmark designed to test AI agents on <span class="highlight">Causal Discovery</span> under adversarial pressure.</p>

            <a href="/docs" class="api-link">View API Documentation (OpenEnv Spec)</a>

            <h2>The Core Mechanic: Phantom Causality</h2>
            <p>Every scenario is backed by a hidden Directed Acyclic Graph (DAG). We plant <strong>Phantom Causes</strong>-highly correlated metrics or config changes that have absolutely nothing to do with the outage.</p>

            <div class="card">
                <h3>Industry-Grade Observability</h3>
                <p>CausalOps supports the 3 pillars of observability to allow true causal discovery:</p>
                <ul>
                    <li><strong>Metrics:</strong> observe('metrics:&lt;svc&gt;') - CPU, Memory, Latency, Error Rate.</li>
                    <li><strong>Logs:</strong> observe('logs:&lt;svc&gt;') - Structured and unstructured application logs.</li>
                    <li><strong>Distributed Tracing:</strong> observe('traces:&lt;svc&gt;') - OpenTelemetry-style spans.</li>
                    <li><strong>GitOps History:</strong> observe('config:&lt;svc&gt;') - Recent deployment history.</li>
                </ul>
            </div>

            <div class="card">
                <h3>Level 2: The Web of Lies (Branching DAG + 1 Phantom)</h3>
                <div class="mermaid">
graph TD
    DB["user-db (Root Cause)"] --> Auth[auth-service]
    Auth --> API[api-gateway]
    Auth --> Pay[payment-gateway]
    DB -.-> DNS["dns-resolver (Phantom Trap!)"]

    style DB fill:#8a2be2,stroke:#fff
    style DNS fill:#b22222,stroke:#fff,stroke-dasharray: 5 5
                </div>
            </div>

            <div class="card">
                <h3>Level 4: The Extreme Mirage (Latent Confounders)</h3>
                <div class="mermaid">
graph TD
    Net["Network Partition (LATENT)"] --> Pay[payment-api]
    Net --> User[user-profile]
    Net -.-> Rec["recommendation-engine (Phantom Config Trap!)"]

    style Net fill:#238636,stroke:#fff
    style Rec fill:#b22222,stroke:#fff,stroke-dasharray: 5 5
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "environment": "causal_ops", "version": "0.1.0"}

def main():
    """Entry point for the server script."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
