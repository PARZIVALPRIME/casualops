"""Baseline evaluation script using OpenAI API. 
Follows strict [START], [STEP], [END] stdout logging.
"""
import os
import json
import argparse

class MockChoice:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})

class MockCompletions:
    def create(self, **kwargs):
        # Always output a valid action to test the loop
        return type("Response", (), {"choices": [MockChoice('{"type": "observe", "target": "metrics:load-balancer", "detail": ""}')]})()

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockOpenAI:
    def __init__(self, api_key=None):
        self.chat = MockChat()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = MockOpenAI

from env.environment import CausalOpsEnvironment
from models import Action, ActionType

def parse_action(text: str) -> Action:
    """Naive parser for OpenAI output to extract the action."""
    # We expect the model to output a JSON block or a line like:
    # ActionType: observe, Target: metrics:database, Detail: 
    # For a baseline, we'll prompt it to output strict JSON.
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(text[start:end])
            return Action(
                type=ActionType(data.get("type", "observe")),
                target=data.get("target", "metrics:load-balancer"),
                detail=data.get("detail", "")
            )
    except Exception:
        pass
    return Action(type=ActionType.OBSERVE, target="metrics:database", detail="")

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent for CausalOps")
    parser.add_argument("--task", type=str, default="easy_smoking_gun")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    # The user must have OPENAI_API_KEY set
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
    
    env = CausalOpsEnvironment()
    
    print("[START]")
    obs = env.reset(args.task)
    
    messages = [
        {"role": "system", "content": (
            "You are a CausalOps Agent. Your job is to diagnose and fix production incidents. "
            "At each step, output a JSON object with 'type', 'target', and 'detail' representing your action.\n"
            "Action types: 'observe', 'hypothesize', 'remediate', 'communicate', 'predict'.\n"
            "Example:\n"
            '{"type": "observe", "target": "metrics:app-server", "detail": ""}\n'
            '{"type": "hypothesize", "target": "database->app-server", "detail": "0.9"}\n'
            '{"type": "remediate", "target": "restart:database", "detail": ""}'
        )}
    ]

    done = False
    step_count = 0
    max_steps = 20

    while not done and step_count < max_steps:
        # Give the agent the current observation
        messages.append({"role": "user", "content": f"Current Observation: {obs.model_dump_json()}"})
        
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.0
            )
            ai_text = response.choices[0].message.content or ""
        except Exception as e:
            # Fallback if no API key or error
            ai_text = '{"type": "observe", "target": "metrics:database", "detail": ""}'

        messages.append({"role": "assistant", "content": ai_text})
        
        action = parse_action(ai_text)
        
        # Step the environment
        step_result = env.step(action)
        
        print(f"[STEP] Action: {action.model_dump_json()} | Reward: {step_result.reward.total}")
        
        obs = step_result.observation
        done = step_result.done
        step_count += 1
        
        if done:
            final_scores = step_result.info.get("final_scores", {})
            print(f"[END] Final Score: {json.dumps(final_scores)}")

if __name__ == "__main__":
    main()
