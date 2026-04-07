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
from fastapi.responses import HTMLResponse

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
            <h1>🔮 CausalOps</h1>
            <p><strong>The Causal Inference Gym for Production Systems</strong></p>
            <p>Welcome to the first OpenEnv benchmark designed to test AI agents on <span class="highlight">Causal Discovery</span> under adversarial pressure.</p>
            
            <a href="/docs" class="api-link">View API Documentation (OpenEnv Spec)</a>

            <h2>The Problem: Confounding Bias</h2>
            <p>Most LLMs suffer from confounding bias. When they see two metrics spike at the same time, they assume causation. CausalOps exploits this by planting <strong>Phantom Causes (Red Herrings)</strong> into the environment using Pearl's Causal Hierarchy.</p>

            <div class="card">
                <h3>Level 2: The Web of Lies (Branching DAG + 1 Phantom)</h3>
                <p>The agent observes API failures and a massive spike in DNS latency. A pattern-matching LLM will try to fix the DNS. A causal-reasoning LLM will realize the DNS spike is just a phantom correlation caused by the User DB load.</p>
                <div class="mermaid">
                graph TD
                    DB[user-db (Root Cause)] --> Auth[auth-service]
                    Auth --> API[api-gateway]
                    Auth --> Pay[payment-gateway]
                    DB -.-> DNS[dns-resolver (Phantom Trap!)]
                    
                    style DB fill:#8a2be2,stroke:#fff
                    style DNS fill:#b22222,stroke:#fff,stroke-dasharray: 5 5
                </div>
            </div>

            <div class="card">
                <h3>Level 4: The Extreme Mirage (Latent Confounders)</h3>
                <p>Multiple independent services fail simultaneously. If the agent tries to fix them, it loses points. The agent must deduce the existence of an unobservable infrastructure failure and escalate to human stakeholders.</p>
                <div class="mermaid">
                graph TD
                    Net[Network Partition (LATENT)] --> Pay[payment-api]
                    Net --> User[user-profile]
                    Net -.-> Rec[recommendation-engine (Phantom Config Trap!)]
                    
                    style Net fill:#238636,stroke:#fff
                    style Rec fill:#b22222,stroke:#fff,stroke-dasharray: 5 5
                </div>
            </div>
            
            <h2>World-Class Features</h2>
            <ul>
                <li><strong>Procedural Topology Generation:</strong> Zero-memorization. The service names are randomly generated using the environment seed.</li>
                <li><strong>Stochastic Jitter:</strong> Metrics contain Gaussian noise.</li>
                <li><strong>Observability Blindspots:</strong> CPU spikes kill telemetry agents.</li>
                <li><strong>GitOps Config History:</strong> Phantoms planted as fake code deployments.</li>
                <li><strong>Dynamic Social Pressure:</strong> Simulated VPs aggressively pressure the agent to fix the wrong service.</li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "environment": "causal_ops", "version": "0.1.0"}
