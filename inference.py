"""
Inference Script for CausalOps OpenEnv Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    API_KEY        Your API key for the LLM proxy.
    MODEL_NAME     The model identifier to use for inference.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

IMPORTANT
- This script uses HTTP requests to talk to the env server (no local imports
  of env/models/graders). This ensures it works when run by the platform's
  evaluation runner, which may not have our local packages installed.
- All LLM calls go through os.environ["API_BASE_URL"] / os.environ["API_KEY"].
"""
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────
# Platform-injected LLM proxy credentials (mandatory)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]

# Our HF Space URL where the env server is running
ENV_URL = os.getenv("ENV_URL", "https://prey7-causal-ops.hf.space")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "causal_ops"
MAX_STEPS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 256

ALL_TASKS = [
    "easy_smoking_gun",
    "medium_web_of_lies",
    "hard_shape_shifter",
    "extreme_latent_mirage",
]

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a CausalOps Agent diagnosing production incidents.
    At each step, output ONLY a JSON object with keys: type, target, detail.

    Action types and targets:
      observe      -> "metrics:<svc>" | "logs:<svc>" | "traces:<svc>" | "config:<svc>"
      hypothesize  -> "<cause>-><effect>"   (detail = confidence 0-1)
      remediate    -> "restart:<svc>" | "scale:<svc>" | "config:<svc>"
      communicate  -> "<stakeholder_id>"    (detail = message text)
      predict      -> "<metric>:<svc>"      (detail = "<expected_delta>,<timeframe_s>")

    Strategy:
    1. First observe metrics/logs for services showing issues.
    2. Hypothesize causal links between services.
    3. Communicate with stakeholders when prompted.
    4. Apply remediation to the root cause service.
    5. Predict the expected metric changes.

    Output ONLY the JSON object, nothing else.
""").strip()


# ── Env HTTP helpers (talk to HF Space, no local imports needed) ──────
def env_reset(task_id: str) -> Dict[str, Any]:
    """POST /reset on the env server, return {observation, reward, done}."""
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: Dict[str, str]) -> Dict[str, Any]:
    """POST /step on the env server, return {observation, reward, done}."""
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Score / action helpers ────────────────────────────────────────────
def clamp_score(v: float) -> float:
    """Clamp score to strictly (0, 1) exclusive range."""
    return round(max(0.01, min(0.99, v)), 2)


def parse_action(text: str) -> Dict[str, str]:
    """Parse LLM output into an action dict. Falls back to a safe default."""
    default = {"type": "observe", "target": "metrics:load-balancer", "detail": ""}
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            return {
                "type": data.get("type", "observe"),
                "target": data.get("target", "metrics:load-balancer"),
                "detail": str(data.get("detail", "")),
            }
    except Exception:
        pass
    return default


# ── Logging helpers ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Main episode runner ───────────────────────────────────────────────
def run_task(task_id: str, client: OpenAI) -> None:
    """Run a single task episode and emit [START]/[STEP]/[END] logs."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    try:
        # Reset env via HTTP
        reset_data = env_reset(task_id)
        obs = reset_data.get("observation", {})
        done = reset_data.get("done", False)

        messages: list = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Build compact observation summary for the LLM
            obs_summary: Dict[str, Any] = {
                "step": obs.get("step_number", step),
                "time_remaining_s": obs.get("time_budget_remaining_s", 0),
                "services": obs.get("services_overview", {}),
                "alerts": obs.get("alerts", [])[:5],
                "logs": obs.get("logs", [])[:10],
                "stakeholder_messages": obs.get("stakeholder_messages", [])[:5],
                "available_actions": obs.get("available_actions", []),
                "task_description": obs.get("task_description", ""),
            }
            if obs.get("detailed_metrics"):
                obs_summary["detailed_metrics"] = obs["detailed_metrics"]
            if obs.get("counterfactual_prompt"):
                obs_summary["counterfactual_prompt"] = obs["counterfactual_prompt"]

            messages.append({"role": "user", "content": json.dumps(obs_summary, default=str)})

            # ── LLM call through the platform proxy ──
            ai_text = '{"type": "observe", "target": "metrics:database", "detail": ""}'
            error_msg = None
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                ai_text = response.choices[0].message.content or ai_text
            except Exception as e:
                error_msg = str(e)

            messages.append({"role": "assistant", "content": ai_text})

            # Parse action and step the env via HTTP
            action = parse_action(ai_text)
            action_str = f"{action['type']}('{action['target']}')"

            try:
                step_data = env_step(action)
                obs = step_data.get("observation", {})
                reward = step_data.get("reward")
                reward = float(reward) if reward is not None else 0.01
                done = step_data.get("done", False)
                steps_taken = step
                rewards.append(reward)

                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            except Exception as e:
                rewards.append(0.01)
                steps_taken = step
                log_step(step=step, action=action_str, reward=0.01, done=True, error=str(e))
                done = True
                break

        # Final score
        if rewards:
            score = rewards[-1] if done else clamp_score(sum(rewards) / len(rewards))
        score = clamp_score(score)

    except Exception as e:
        score = 0.01
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    # Initialize OpenAI client with platform-injected proxy credentials
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    single_task = os.getenv("CAUSAL_OPS_TASK")
    if single_task:
        tasks = [single_task]
    else:
        tasks = ALL_TASKS

    for task_id in tasks:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
