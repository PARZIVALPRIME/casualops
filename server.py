"""FastAPI layer for CausalOps OpenEnv environment.

Thin HTTP wrapper — all logic lives in env/environment.py.
Endpoints:
  POST /reset          — start a new episode
  POST /step           — submit an action
  GET  /state          — get full state
  GET  /tasks          — list available tasks
  GET  /health         — health check
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env import CausalOpsEnvironment
from models import Action, Observation, State, StepResult

from fastapi.middleware.cors import CORSMiddleware

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


# ── Endpoints ────────────────────────────────────────────────────
@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    """Initialize a new episode for the given task."""
    try:
        obs = _env.reset(req.task_id, seed=req.seed)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """Execute one agent action and advance the environment."""
    try:
        result = _env.step(action)
        return result
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


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "environment": "causal_ops", "version": "0.1.0"}
