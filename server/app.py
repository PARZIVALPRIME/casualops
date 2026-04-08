"""FastAPI application for CausalOps OpenEnv environment.

Uses openenv.core.env_server.create_app for spec compliance.
A single shared environment instance is used since this env is stateful
and HTTP requests must share state across reset/step/state calls.
"""
from openenv.core.env_server import create_app

from env.environment import CausalOpsEnvironment
from models import CausalOpsAction, CausalOpsObservation

# Singleton: all HTTP requests share one environment instance
_shared_env = CausalOpsEnvironment()

app = create_app(lambda: _shared_env, CausalOpsAction, CausalOpsObservation, env_name="causal_ops")

def main():
    """Entry point for the server script."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
