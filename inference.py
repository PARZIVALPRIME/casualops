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
"""
import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from client import CausalOpsClient
from models import CausalOpsAction, ActionType

# ── Configuration ──────────────────────────────────────────────────────
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


def clamp_score(v: float) -> float:
    return round(max(0.01, min(0.99, v)), 2)


def parse_action(text: str) -> CausalOpsAction:
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
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Async episode runner ──────────────────────────────────────────────
async def run_task(task_id: str, llm: OpenAI, env: CausalOpsClient) -> None:
    """Run a single task episode using async env client."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        done = result.done

        messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            obs_dict: Dict[str, Any] = {
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
                obs_dict["detailed_metrics"] = {
                    k: {"cpu": v.cpu_percent, "latency_ms": v.latency_ms,
                         "error_rate": v.error_rate, "status": v.status}
                    for k, v in obs.detailed_metrics.items()
                }
            if obs.counterfactual_prompt:
                obs_dict["counterfactual_prompt"] = obs.counterfactual_prompt.message

            messages.append({"role": "user", "content": json.dumps(obs_dict, default=str)})

            # ── LLM call through the platform proxy ──
            ai_text = '{"type": "observe", "target": "metrics:database", "detail": ""}'
            error_msg = None
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                ai_text = response.choices[0].message.content or ai_text
            except Exception as e:
                error_msg = str(e)

            messages.append({"role": "assistant", "content": ai_text})

            action = parse_action(ai_text)
            action_str = f"{action.type.value}('{action.target}')"

            try:
                result = await env.step(action)
                obs = result.observation
                reward = float(result.reward) if result.reward is not None else 0.01
                done = result.done
                steps_taken = step
                rewards.append(reward)
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            except Exception as e:
                rewards.append(0.01)
                steps_taken = step
                log_step(step=step, action=action_str, reward=0.01, done=True, error=str(e))
                done = True
                break

        if rewards:
            score = rewards[-1] if done else clamp_score(sum(rewards) / len(rewards))
        score = clamp_score(score)

    except Exception as e:
        score = 0.01
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    # LLM client — must use platform-injected proxy
    llm = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"],
    )

    tasks = [os.environ["CAUSAL_OPS_TASK"]] if os.getenv("CAUSAL_OPS_TASK") else ALL_TASKS

    # Start env from Docker image, passing PORT=8000 so the container
    # binds to port 8000 (OpenEnv's LocalDockerProvider default mapping)
    env = await CausalOpsClient.from_docker_image(
        "casualops:latest",
        env_vars={"PORT": "8000"},
    )

    try:
        for task_id in tasks:
            await run_task(task_id, llm, env)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
