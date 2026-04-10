import os
import textwrap
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from models import ClinicAction
from server.clinic_scheduler_environment import ClinicSchedulerEnvironment

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 8
TEMPERATURE = 0.2
MAX_TOKENS = 32

TASKS = {
    "easy": "Low demand clinic with mild variability.",
    "medium": "Balanced clinic with a midday demand surge.",
    "hard": "High demand clinic with strong peak-hour pressure.",
}


class TaskGrader:
    @staticmethod
    def score_episode(task: str, final_obs, rewards: list[float], steps_taken: int) -> float:
        if not rewards:
            return 0.01

        min_score = 0.01
        max_score = 0.99

        avg_reward = sum(rewards) / len(rewards)
        total_no_shows = final_obs.info.get("total_no_shows", 0)
        total_served = final_obs.info.get("total_served", 0)

        final_walk_in_queue = getattr(final_obs, "walk_in_queue", 0)
        final_reserved_queue = getattr(final_obs, "reserved_queue", 0)
        final_backlog = final_walk_in_queue + final_reserved_queue

        if task == "easy":
            served_score = min(1.0, total_served / 18.0)
            backlog_score = max(0.0, 1.0 - final_backlog / 5.0)
            reward_score = max(0.0, min(1.0, (avg_reward + 4.0) / 4.0))
            no_show_score = max(0.0, 1.0 - total_no_shows / 3.0)
            raw = 0.25 * served_score + 0.35 * backlog_score + 0.25 * reward_score + 0.15 * no_show_score

        elif task == "medium":
            served_score = min(1.0, total_served / 24.0)
            backlog_score = max(0.0, 1.0 - final_backlog / 8.0)
            reward_score = max(0.0, min(1.0, (avg_reward + 5.0) / 5.0))
            no_show_score = max(0.0, 1.0 - total_no_shows / 5.0)
            raw = 0.20 * served_score + 0.40 * backlog_score + 0.25 * reward_score + 0.15 * no_show_score

        else:
            served_score = min(1.0, total_served / 30.0)
            backlog_score = max(0.0, 1.0 - final_backlog / 8.0)
            reward_score = max(0.0, min(1.0, (avg_reward + 6.0) / 6.0))
            no_show_score = max(0.0, 1.0 - total_no_shows / 7.0)
            raw = 0.10 * served_score + 0.50 * backlog_score + 0.25 * reward_score + 0.15 * no_show_score

        if steps_taken == MAX_STEPS:
            raw += 0.02

        return max(min_score, min(max_score, raw))
    
    @staticmethod
    def threshold(task: str) -> float:
        return {
            "easy": 0.60,
            "medium": 0.52,
            "hard": 0.45,
        }[task]

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} model={model}", flush=True)


def log_step(step: int, action: float, reward: float, done: bool) -> None:
    print(
        f"[STEP] step={step} action=walk_in_ratio={action:.2f} reward={reward:.3f} done={str(done).lower()}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} rewards=[{rewards_str}] grader_score={score:.4f}",
        flush=True,
    )


def build_prompt(task: str, step: int, obs) -> str:
    return textwrap.dedent(
        f"""
        Environment: clinic scheduling
        Difficulty: {task}
        Hour: {step}/8

        Current state:
        - Walk-in queue: {obs.walk_in_queue}
        - Reserved queue: {obs.reserved_queue}
        - Walk-in slots last hour: {obs.walk_in_slots}
        - Reserved slots last hour: {obs.reserved_slots}

        Goal:
        Choose a walk-in capacity ratio between 0.1 and 0.9 to reduce queue buildup and no-show waste.

        Reply with only one decimal number between 0.1 and 0.9.
        """
    ).strip()


def get_model_action(client: OpenAI, task: str, step: int, obs) -> float:
    prompt = build_prompt(task, step, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an RL policy for a clinic scheduling environment. Output only a number."},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        ratio = float(text)
        return max(0.1, min(0.9, ratio))
    except Exception:
        return 0.5


def run_task(task: str, client: OpenAI) -> Tuple[bool, int, List[float], float]:
    env = ClinicSchedulerEnvironment(task=task, seed=42)
    log_start(task, MODEL_NAME)

    obs = env.reset(seed=42)
    rewards: List[float] = []
    steps_taken = 0

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        action_value = get_model_action(client, task, step, obs)
        action = ClinicAction(walk_in_ratio=action_value)

        obs = env.step(action)
        reward = obs.reward or 0.0
        rewards.append(reward)
        steps_taken = step

        log_step(step, action_value, reward, obs.done)

        if obs.done:
            break

    score = TaskGrader.score_episode(task, obs, rewards, steps_taken)
    success = score >= TaskGrader.threshold(task)
    log_end(task, success, steps_taken, rewards, score)

    return success, steps_taken, rewards, score


def main() -> None:
    if not API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = {}
    for task in ["easy", "medium", "hard"]:
        success, steps, rewards, score = run_task(task, client)
        results[task] = {"success": success, "steps": steps, "rewards": rewards, "score": score}

    print("\n" + "=" * 60)
    print("TASK SUMMARY")
    print("=" * 60)

    total = 0.0
    for task in ["easy", "medium", "hard"]:
        result = results[task]
        avg_reward = sum(result["rewards"]) / len(result["rewards"]) if result["rewards"] else 0.0
        status = "[PASS]" if result["success"] else "[FAIL]"
        print(f"{task.upper():8} {status:8} score={result['score']:.4f} avg_reward={avg_reward:+.3f} steps={result['steps']}/8")
        total += result["score"]

    print("=" * 60)
    print(f"Average Score: {total / 3.0:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()