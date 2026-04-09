"""
Inference Script for CausalOps OpenEnv Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - Each tasks should return score in (0, 1) exclusive.
"""
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import CausalOpsClient
from models import CausalOpsAction, ActionType

# ── Configuration ──────────────────────────────────────────────────────
# IMPORTANT: The competition platform injects API_BASE_URL and API_KEY
# to route through their LiteLLM proxy. We MUST use these first.
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "causal_ops"
MAX_STEPS = 20
TEMPERATURE = 0.0
MAX_TOKENS = 256

# Tasks to run — platform sets CAUSAL_OPS_TASK for single-task runs
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


def clamp_score(v: float) -> float:
    """Clamp score to strictly (0, 1) exclusive range."""
    return round(max(0.01, min(0.99, v)), 2)


def parse_action(text: str) -> CausalOpsAction:
    """Parse LLM output into an action. Falls back to a safe default."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            return CausalOpsAction(
                type=ActionType(data.get("type", "observe")),
                target=data.get("target", "metrics:load-balancer"),
                detail=str(data.get("detail", "")),
            )
    except Exception:
        pass
    return CausalOpsAction(type=ActionType.OBSERVE, target="metrics:load-balancer", detail="")


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


def run_task(task_id: str, client: OpenAI) -> None:
    """Run a single task episode and emit [START]/[STEP]/[END] logs."""
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01  # safe default in (0, 1)
    success = False

    try:
        with CausalOpsClient().sync() as env:
            # Reset with correct keyword argument
            result = env.reset(task_id=task_id)
            obs = result.observation

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            done = result.done

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                # Build user prompt from current observation
                obs_summary = {
                    "step": obs.step_number,
                    "time_remaining_s": round(obs.time_budget_remaining_s, 1),
                    "services": obs.services_overview,
                    "alerts": obs.alerts[:5],
                    "logs": obs.logs[:10],
                    "stakeholder_messages": [
                        {"sender": m.sender, "message": m.message, "requires_response": m.requires_response}
                        for m in obs.stakeholder_messages
                    ],
                    "available_actions": obs.available_actions,
                    "task_description": obs.task_description,
                }
                if obs.detailed_metrics:
                    obs_summary["detailed_metrics"] = {
                        k: {"cpu": v.cpu_percent, "latency_ms": v.latency_ms,
                             "error_rate": v.error_rate, "status": v.status}
                        for k, v in obs.detailed_metrics.items()
                    }
                if obs.counterfactual_prompt:
                    obs_summary["counterfactual_prompt"] = obs.counterfactual_prompt.message

                messages.append({"role": "user", "content": json.dumps(obs_summary, default=str)})

                # Get LLM action
                ai_text = '{"type": "observe", "target": "metrics:database", "detail": ""}'
                error_msg = None
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,  # type: ignore
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    ai_text = response.choices[0].message.content or ai_text
                except Exception as e:
                    error_msg = str(e)

                messages.append({"role": "assistant", "content": ai_text})

                # Parse and execute action
                action = parse_action(ai_text)
                action_str = f"{action.type.value}('{action.target}')"

                try:
                    # env.step() returns StepResult
                    result = env.step(action)
                    obs = result.observation
                    reward = result.reward if result.reward is not None else 0.01
                    reward = float(reward)
                    done = result.done
                    steps_taken = step
                    rewards.append(reward)

                    log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
                    
                    if done:
                        state_res = env.state()
                        success = state_res.state.get("remediation_successful", False) if state_res.state else False

                except Exception as e:
                    rewards.append(0.01)
                    steps_taken = step
                    log_step(step=step, action=action_str, reward=0.01, done=True, error=str(e))
                    done = True
                    break

            # Compute final score from the last reward (which is the final grader score when done)
            if rewards:
                score = rewards[-1] if done else clamp_score(sum(rewards) / len(rewards))
            score = clamp_score(score)

    except Exception as e:
        # Even on total failure, emit a valid [END] with score in (0, 1)
        score = 0.01
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    # Set default environment variables for local testing,
    # so we don't crash when strictly using os.environ[] below.
    if "API_BASE_URL" not in os.environ:
        os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
    if "API_KEY" not in os.environ:
        os.environ["API_KEY"] = os.environ.get("HF_TOKEN", "dummy")
    
    # Debug: confirm we're using the platform's proxy
    print(f"[DEBUG] API_BASE_URL={os.environ['API_BASE_URL']}", flush=True)
    print(f"[DEBUG] API_KEY={'***' + os.environ['API_KEY'][-4:] if len(os.environ['API_KEY']) > 4 else '(short)'}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    # Initialize your OpenAI client EXACTLY as the validator requires:
    client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])

    # Support single-task via env var (platform sets this) or run all tasks
    single_task = os.getenv("CAUSAL_OPS_TASK")
    if single_task:
        tasks = [single_task]
    else:
        tasks = ALL_TASKS

    for task_id in tasks:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
