"""
Clinic Scheduler Inference Script with Task Graders
=====================================================
Runs 3 graded tasks (easy, medium, hard) with OpenEnv compliance.
Each task has a programmatic grader that scores 0.0 < score < 1.0.

MANDATORY ENVIRONMENT VARIABLES:
    OPENAI_API_KEY  Your OpenAI API key (or HF token if using HF router)
    API_BASE_URL    (optional) API endpoint (default: HF router)
    MODEL_NAME      (optional) Model to use (default: Qwen/Qwen2.5-72B-Instruct)
"""

import os
import textwrap
from typing import List, Tuple
from dotenv import load_dotenv

from openai import OpenAI
from client import ClinicEnv, ClinicAction
from server.clinic_scheduler_environment import ClinicSchedulerEnvironment

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = "clinic-scheduling"
BENCHMARK = "clinic_scheduler"
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 100

# Task Definitions
TASKS = {
    "easy": {
        "description": "Low demand (0.5-2.0 arrivals/hour). Simple ratio decisions.",
        "system_prompt": textwrap.dedent("""
            You are scheduling a low-demand clinic (0.5-2 patients/hour).
            This is a BASIC TASK: Simple decisions, minimal complexity.
            
            Task Objective: Minimize wait times and no-shows.
            - Walk-in ratio: 0.1-0.9 (decide EACH hour)
            - Only 1 patient/hour treatment capacity (bottleneck!)
            - Reward = 3.0 - (avg_wait + no_shows)
            - Perfect = +3.0 (no wait, no no-shows)
            
            Strategy: Balanced ratio (0.4-0.6) works well here.
            Output: Only a decimal number 0.1-0.9 (the ratio), nothing else.
        """).strip(),
    },
    "medium": {
        "description": "Balanced demand with peak surge (hours 3-5). Requires strategic planning.",
        "system_prompt": textwrap.dedent("""
            You are scheduling a clinic with PEAK SURGE (hours 3-5).
            This is a MODERATE TASK: Requires strategic anticipation.
            
            Task Objective: Manage peak hours effectively.
            - Normal hours: 1-4 patients/hour
            - PEAK hours (3-5): 4-7 patients/hour SURGE!
            - Only 1 patient/hour treatment capacity
            - Reward = 3.0 - (avg_wait + no_shows)
            - Perfect = +3.0
            - Bad peak decisions = NEGATIVE rewards
            
            Strategy: 
            - Hours 1-2: Conservative (0.3-0.4 ratio)
            - Hours 3-5: Aggressive (0.7-0.9 ratio) to handle surge
            - Hours 6-8: Moderate (0.5-0.6 ratio)
            
            Output: Only decimal 0.1-0.9, nothing else.
        """).strip(),
    },
    "hard": {
        "description": "High demand with aggressive peak surge. Demanding optimization required.",
        "system_prompt": textwrap.dedent("""
            You are scheduling a HIGH-DEMAND clinic with AGGRESSIVE SURGE.
            This is a HARD TASK: Requires optimal strategic planning.
            
            Task Objective: Navigate extreme conditions optimally.
            - Normal hours: 2-5 patients/hour (high baseline)
            - PEAK hours (3-5): 5.5-9 patients/hour AGGRESSIVE SURGE!
            - Only 1 patient/hour treatment capacity (critical bottleneck!)
            - Reward = 3.0 - (avg_wait + no_shows)
            - Perfect = +3.0
            - Poor decisions = VERY NEGATIVE rewards (-2 to -4)
            
            Challenge: Queue buildup is inevitable. Minimize damage.
            Strategy:
            - Hours 1-2: Very low ratio (0.2-0.3) to prepare reserves
            - Hours 3-5: Maximum walk-in (0.85-0.95) for surge
            - Hours 6-8: High ratio (0.7-0.8) to clear queue
            
            Output: Only decimal 0.1-0.9, nothing else.
        """).strip(),
    },
}


class TaskGrader:
    """Grades task performance with scores strictly in (0.0, 1.0)."""
    
    @staticmethod
    def grade(task: str, rewards: List[float]) -> float:
        """Score performance: 0.0 < score < 1.0.
        
        Args:
            task: "easy", "medium", or "hard"
            rewards: List of step rewards
            
        Returns:
            Score in range (0.0, 1.0)
        """
        if not rewards:
            return 0.1  # Minimum non-zero score
        
        avg_reward = sum(rewards) / len(rewards)
        total_reward = sum(rewards)
        
        # Task-specific scoring with baseline 3.0
        if task == "easy":
            # Easy: expect higher avg rewards (baseline 2.0-3.0)
            # Map: avg_reward=0 → 0.15, avg_reward=1.5 → 0.5, avg_reward=2.5 → 0.88
            normalized = (avg_reward + 1.0) / 3.5
            score = max(0.1, min(0.95, 0.15 + 0.75 * normalized))
        elif task == "medium":
            # Medium: balanced rewards (baseline 1.0-2.5)
            # Map: avg_reward=-0.5 → 0.2, avg_reward=1.0 → 0.55, avg_reward=2.0 → 0.85
            normalized = (avg_reward + 0.5) / 2.5
            score = max(0.15, min(0.9, 0.2 + 0.65 * normalized))
        else:  # hard
            # Hard: lower expected rewards (baseline -0.5 to 2.0)
            # Map: avg_reward=-3 → 0.1, avg_reward=0 → 0.4, avg_reward=1.5 → 0.8
            normalized = (avg_reward + 3.0) / 4.5
            score = max(0.1, min(0.95, 0.1 + 0.8 * normalized))
        
        # Bonus for completing all steps without failure
        if len(rewards) == MAX_STEPS:
            score += 0.02  # Small completion bonus
            score = min(0.95, score)  # Cap at 0.95
        
        # Ensure strictly within (0.0, 1.0)
        return max(0.01, min(0.99, score))

    @staticmethod
    def get_threshold(task: str) -> float:
        """Success threshold for each task (strictly between 0-1)."""
        thresholds = {
            "easy": 0.50,    # 50% score needed (easy = simple)
            "medium": 0.48,  # 48% score needed
            "hard": 0.35,    # 35% score needed
        }
        return thresholds.get(task, 0.5)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    success_str = str(success).lower()
    print(
        f"[END] task={task} success={success_str} steps={steps} rewards=[{rewards_str}] grader_score={score:.4f}",
        flush=True,
    )


def build_user_prompt(
    step: int,
    patients_waiting: float,
    walk_in_slots: int,
    reserved_slots: int,
    last_reward: float,
    hour_context: str,
) -> str:
    """Build prompt for LLM decision."""
    return textwrap.dedent(
        f"""
        ===== HOUR {step} / 8 =====
        Clinic Status:
          - Patients waiting: {patients_waiting:.1f}
          - Walk-in slots available: {walk_in_slots}
          - Reserved slots available: {reserved_slots}
          - Last hour reward: {last_reward:.2f}
        
        {hour_context}
        
        DECISION: Choose walk-in ratio for THIS HOUR (0.1-0.9)
        Output ONLY the number, nothing else.
        """
    ).strip()


def get_model_action_with_system(
    client: OpenAI,
    system_prompt: str,
    step: int,
    patients_waiting: float,
    walk_in_slots: int,
    reserved_slots: int,
    last_reward: float,
) -> float:
    """Get action from LLM with system prompt."""
    hour_context = {
        1: "Hour 1 of 8. Starting hour.",
        2: "Hour 2 of 8. Demand ramping up.",
        3: "⚠️ PEAK SURGE STARTS! High arrivals expected.",
        4: "🔴 PEAK SURGE CONTINUES! Heavy traffic.",
        5: "🔴 PEAK SURGE ENDING! Still intense.",
        6: "📉 Back to normal demand. Clear queues.",
        7: "Hour 7. Almost done. Final push.",
        8: "Hour 8. FINAL HOUR. End of shift.",
    }.get(step, "Hour X/8.")
    
    user_prompt = build_user_prompt(step, patients_waiting, walk_in_slots, reserved_slots, last_reward, hour_context)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        try:
            ratio = float(text)
            ratio = max(0.1, min(0.9, ratio))
            return ratio
        except ValueError:
            print(f"[DEBUG] Failed to parse ratio from model: {text!r}, using default 0.5", flush=True)
            return 0.5
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {type(exc).__name__}: {exc}", flush=True)
        return 0.5


def run_task(task: str, client: OpenAI) -> Tuple[bool, int, List[float], float]:
    """Run a single task and return (success, steps, rewards, score)."""
    # Create environment directly with task config
    env_instance = ClinicSchedulerEnvironment(task=task)
    
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    system_prompt = TASKS[task]["system_prompt"]
    
    try:
        obs = env_instance.reset()
        last_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            
            # Get LLM action with task-specific system prompt
            walk_in_ratio = get_model_action_with_system(
                client,
                system_prompt,
                step,
                obs.patients_waiting,
                obs.walk_in_slots,
                obs.reserved_slots,
                last_reward,
            )
            
            action = ClinicAction(walk_in_ratio=walk_in_ratio)
            obs = env_instance.step(action)
            
            reward = obs.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            action_str = f"walk_in_ratio={walk_in_ratio:.2f}"
            log_step(step=step, action=action_str, reward=reward, done=obs.done)
            
            if obs.done:
                break
        
        grader = TaskGrader()
        score = grader.grade(task, rewards)
        threshold = grader.get_threshold(task)
        success = score >= threshold
        
    except Exception as e:
        print(f"[DEBUG] Error during task {task}: {e}", flush=True)
        score = 0.05  # Minimum score on error
        success = False
    finally:
        grader = TaskGrader()
        score = grader.grade(task, rewards)
        log_end(task=task, success=success, steps=steps_taken, rewards=rewards, score=score)
    
    return success, steps_taken, rewards, score


def main() -> None:
    """Run all 3 graded tasks."""
    print(f"[DEBUG] Configuration:", flush=True)
    print(f"[DEBUG]   API_KEY present: {bool(API_KEY)}", flush=True)
    print(f"[DEBUG]   API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG]   MODEL_NAME: {MODEL_NAME}", flush=True)
    print(f"[DEBUG]", flush=True)
    
    if not API_KEY:
        print("[ERROR] OPENAI_API_KEY not set. Please set it in .env or environment.", flush=True)
        return
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Run all 3 tasks
    task_results = {}
    for task_name in ["easy", "medium", "hard"]:
        print(f"\n[INFO] Starting task: {task_name}", flush=True)
        print(f"[INFO] Description: {TASKS[task_name]['description']}", flush=True)
        
        success, steps, rewards, score = run_task(task_name, client)
        task_results[task_name] = {
            "success": success,
            "steps": steps,
            "rewards": rewards,
            "score": score,
        }
    
    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"TASK SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    
    total_score = 0.0
    for task_name in ["easy", "medium", "hard"]:
        result = task_results[task_name]
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        avg_reward = sum(result["rewards"]) / len(result["rewards"]) if result["rewards"] else 0.0
        print(
            f"{task_name.upper():8} {status:8} score={result['score']:.4f} "
            f"avg_reward={avg_reward:+.2f} steps={result['steps']}/8"
        )
        total_score += result["score"]
    
    avg_task_score = total_score / 3.0
    print(f"{'='*60}", flush=True)
    print(f"Average Score: {avg_task_score:.4f} (range 0.0-1.0)", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

