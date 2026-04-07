"""Baseline evaluation script using OpenAI API. 
Follows strict [START], [STEP], [END] stdout logging.
"""
import os
import json
import argparse
from typing import Optional

from openai import OpenAI

from env.environment import CausalOpsEnvironment
from models import Action, ActionType

def parse_action(text: str) -> Action:
    """Naive parser for OpenAI output to extract the action."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(text[start:end])
            return Action(
                type=ActionType(data.get("type", "observe")),
                target=data.get("target", "metrics:load-balancer"),
                detail=str(data.get("detail", ""))
            )
    except Exception:
        pass
    return Action(type=ActionType.OBSERVE, target="metrics:load-balancer", detail="")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent for CausalOps")
    parser.add_argument("--task", type=str, default="easy_smoking_gun")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "dummy")
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = str(os.environ.get("MODEL_NAME", args.model))
    
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)  # type: ignore
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    env = CausalOpsEnvironment()
    
    log_start(task=args.task, env="causal_ops", model=model_name)
    
    try:
        obs = env.reset(args.task)
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return
    
    messages = [
        {"role": "system", "content": (
            "You are a CausalOps Agent. Your job is to diagnose and fix production incidents.\n"
            "At each step, output a JSON object with 'type', 'target', and 'detail' representing your action.\n"
            "Action types: 'observe', 'hypothesize', 'remediate', 'communicate', 'predict'.\n"
            "Example:\n"
            '{"type": "observe", "target": "metrics:app-server", "detail": ""}\n'
            '{"type": "hypothesize", "target": "database->app-server", "detail": "0.9"}\n'
            '{"type": "remediate", "target": "restart:database", "detail": ""}\n'
            'Only output the JSON object, nothing else.'
        )}
    ]

    done = False
    step_count = 0
    max_steps = 20
    rewards_history = []
    success = False
    score = 0.0

    while not done and step_count < max_steps:
        step_count += 1
        messages.append({"role": "user", "content": f"Current Observation: {obs.model_dump_json()}"})
        
        ai_text = '{"type": "observe", "target": "metrics:database", "detail": ""}'
        error_msg = None
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0
            )
            ai_text = response.choices[0].message.content or ""
        except Exception as e:
            error_msg = str(e)
            
        messages.append({"role": "assistant", "content": ai_text})
        
        action = parse_action(ai_text)
        action_str = f"{action.type.value}('{action.target}')"
        
        try:
            step_result = env.step(action)
            reward_val = step_result.reward.total
            done = step_result.done
            obs = step_result.observation
            rewards_history.append(reward_val)
            
            if done and "final_scores" in step_result.info:
                scores_dict = step_result.info["final_scores"]
                if isinstance(scores_dict, dict):
                    score = scores_dict.get("total", 0.0)
                success = env.state().remediation_successful
                
            log_step(step=step_count, action=action_str, reward=reward_val, done=done, error=error_msg)
            
        except Exception as e:
            rewards_history.append(0.0)
            log_step(step=step_count, action=action_str, reward=0.0, done=True, error=str(e))
            break
            
    log_end(success=success, steps=step_count, score=score, rewards=rewards_history)

if __name__ == "__main__":
    main()
