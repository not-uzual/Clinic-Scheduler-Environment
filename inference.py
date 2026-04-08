"""
Clinic Scheduler Inference Script
===================================
MANDATORY ENVIRONMENT VARIABLES:
    API_KEY         Your Hugging Face / API key (from .env or environment)
    API_BASE_URL    The API endpoint for the LLM (default: HF router)
    MODEL_NAME      The model identifier to use for inference
    LOCAL_IMAGE_NAME The name of the local Docker image (e.g., clinic_scheduler-env:latest)

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

from openai import OpenAI
from client import ClinicEnv, ClinicAction

# Load environment variables from .env file
load_dotenv()

# Environment configuration
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "clinic_scheduler-env:latest")

TASK_NAME = os.getenv("TASK_NAME", "clinic-scheduling")
BENCHMARK = os.getenv("BENCHMARK", "clinic_scheduler")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 100

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an intelligent agent managing a busy clinic's appointment scheduling in real-time.
    
    Context - CHALLENGING ENVIRONMENT:
    - Total slots available per hour: 10
    - You decide the walk-in ratio (0.1 to 0.9) EACH HOUR
    - Example: 0.6 = 60% walk-in (6 slots), 40% reserved (4 slots)
    - Patient arrivals: 
      * Normal hours (1-2, 6-8): 1-4 per hour
      * PEAK SURGE (hours 3-5): 4-7 per hour (!!)
    - Treatment capacity: ONLY 1 patient per hour (bottleneck!)
    - No-shows are HEAVILY penalized (1.0x multiplier)
    - Reward = 3 - (avg_wait_time + no_shows)
    - PERFECT performance = +3.0 reward (no wait, no no-shows)
    - AVERAGE performance = 0 to +1.0 reward
    - BAD decisions = NEGATIVE rewards (will hurt badly!)
    
    Strategic Challenge:
    - Low ratio (0.2-0.3): More reserved slots = more no-shows + severe penalty during surge
    - High ratio (0.7-0.9): More walk-in = less no-shows but must manage instability during surge
    - PEAK HOURS (3-5) will PUNISH wrong decisions with highly NEGATIVE rewards!
    
    Anticipate peak demand. Bad planning = real pain.
    
    Output format: Only output a decimal number between 0.1 and 0.9 representing this hour's walk_in ratio.
    No explanations, no quotes, just the number.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def build_user_prompt(
    step: int, 
    patients_waiting: int, 
    walk_in_slots: int,
    reserved_slots: int,
    last_reward: float,
    history: List[str]
) -> str:
    history_block = "\n".join(history[-3:]) if history else "None"
    
    # Warn about upcoming peak hours
    upcoming = ""
    if step <= 2:
        upcoming = "\n⚠️  WARNING: Peak demand surge coming in hours 3-5 (4-7 arrivals/hour)!"
    elif step in [3, 4, 5]:
        upcoming = "\n🔴 PEAK HOUR NOW: High arrivals expected (4-7/hour). Adjust ratio strategically!"
    elif step > 5:
        upcoming = "\n📉 Peak surge has passed. Manage backlog in remaining hours."
    
    return textwrap.dedent(
        f"""
        ===== HOUR {step} / 8 =====
        Clinic Status:
          - Patients waiting to be treated: {patients_waiting}
          - Current walk-in slots: {walk_in_slots}
          - Current reserved slots: {reserved_slots}
          - Last hour reward: {last_reward:.2f}
        
        Recent History:
        {history_block}{upcoming}
        
        DECISION: Choose walk-in ratio for THIS HOUR (0.1 to 0.9)
        - High ratio → fewer no-shows, handles surge better
        - Low ratio → more no-shows (COSTLY!), worse during surge
        - Only 1 patient/hour capacity means backlog builds fast
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    step: int,
    patients_waiting: int,
    walk_in_slots: int,
    reserved_slots: int,
    last_reward: float,
    history: List[str]
) -> float:
    user_prompt = build_user_prompt(step, patients_waiting, walk_in_slots, reserved_slots, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        try:
            ratio = float(text)
            ratio = max(0.1, min(0.9, ratio))  # Clamp to valid range
            return ratio
        except ValueError:
            print(f"[DEBUG] Failed to parse ratio from model: {text!r}, using default 0.5", flush=True)
            return 0.5
    except Exception as exc:
        import traceback
        print(f"[DEBUG] Model request failed: {type(exc).__name__}", flush=True)
        print(f"[DEBUG] Error details: {str(exc)}", flush=True)
        print(f"[DEBUG] Traceback:", flush=True)
        traceback.print_exc()
        return 0.5


def main() -> None:
    print(f"[DEBUG] Configuration loaded:", flush=True)
    print(f"[DEBUG]   API_KEY present: {bool(API_KEY)}", flush=True)
    print(f"[DEBUG]   API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG]   MODEL_NAME: {MODEL_NAME}", flush=True)
    
    if not API_KEY:
        print("[ERROR] API_KEY not set. Please set it in .env or environment.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Create environment from Docker image
    try:
        env = ClinicEnv.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to create environment from Docker: {e}", flush=True)
        return

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    error_msg: Optional[str] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        result = env.reset()
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # Get action from model
            walk_in_ratio = get_model_action(
                client,
                step,
                obs.patients_waiting,
                obs.walk_in_slots,
                obs.reserved_slots,
                last_reward,
                history
            )

            # Take step in environment
            action = ClinicAction(walk_in_ratio=walk_in_ratio)
            result = env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = obs.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            action_str = f"walk_in_ratio={walk_in_ratio:.2f}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            history.append(f"Hour {step}: ratio={walk_in_ratio:.2f} → reward {reward:+.2f}")

            if done:
                break

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success = avg_reward > -5.0  # Accept if not catastrophically bad

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Error during episode: {e}", flush=True)
        success = False
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
