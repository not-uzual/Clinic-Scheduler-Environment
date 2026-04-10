import math
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
TEMPERATURE = 0.7
MAX_TOKENS = 256

TASKS = {
    "easy": "Low demand clinic with mild variability.",
    "medium": "Balanced clinic with a midday demand surge.",
    "hard": "High demand clinic with strong peak-hour pressure.",
}


class TaskGrader:
    _EPS = 0.01

    @staticmethod
    def score_episode(task: str, final_obs, rewards: list[float], steps_taken: int) -> float:
        if not rewards:
            return TaskGrader._EPS

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

        # Sigmoid squash — mathematically impossible to return exactly 0.0 or 1.0
        sigmoid = 1.0 / (1.0 + math.exp(-4.0 * (raw - 0.5)))
        return min(max(sigmoid, TaskGrader._EPS), 1.0 - TaskGrader._EPS)

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


def build_system_prompt() -> str:
    return textwrap.dedent("""
        You are an RL policy managing a medical clinic scheduler.

        Each hour you must choose a walk_in_ratio (0.1 to 0.9) that splits 10 slots:
        - walk_in_slots = round(10 * ratio)
        - reserved_slots = 10 - walk_in_slots

        Reward signals:
        - Serving patients earns +0.18 each
        - Reserved queue backlog costs -0.42 each (they had appointments!)
        - Walk-in queue backlog costs -0.28 each
        - No-shows cost -0.60 each (wasted reserved slot)
        - Idle slots cost -0.25 each
        - Mismatched allocation costs -0.18

        Strategy:
        - If reserved_queue > walk_in_queue, lower the ratio (give more slots to reserved)
        - If walk_in_queue > reserved_queue, raise the ratio (give more slots to walk-ins)
        - During peak hours (hour 2-4), demand spikes so avoid idle slots
        - No-shows waste reserved slots, so don't over-allocate to reserved
        - NEVER pick 0.5 blindly. Always justify based on queue imbalance.
        - If walk_in_queue == 0 and reserved_queue > 0: pick 0.3 or lower
        - If reserved_queue == 0 and walk_in_queue > 0: pick 0.7 or higher
        - If both queues are 0: pick 0.4 (expect more walk-ins than reserved)
        - If queues are equal: pick 0.5 only then

        Think step by step about the queue sizes and last reward. Then on the final line output only a single decimal number between 0.1 and 0.9.
    """).strip()


def build_prompt(task: str, step: int, obs, last_reward: float | None) -> str:
    reward_line = (
        f"- Last step reward: {last_reward:.3f}"
        if last_reward is not None
        else "- Last step reward: N/A (first step)"
    )
    return textwrap.dedent(
        f"""
        Difficulty: {task} | Hour: {step}/8

        Current state:
        - Walk-in queue: {obs.walk_in_queue}
        - Reserved queue: {obs.reserved_queue}
        - Walk-in slots last hour: {obs.walk_in_slots}
        - Reserved slots last hour: {obs.reserved_slots}
        - Total served so far: {obs.info.get('total_served', 0)}
        - Total no-shows so far: {obs.info.get('total_no_shows', 0)}
        - Cumulative wait cost: {obs.info.get('cumulative_wait_cost', 0)}
        {reward_line}

        What walk_in_ratio do you choose? (0.1 to 0.9)
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    task: str,
    step: int,
    obs,
    history: list,
    last_reward: float | None,
) -> tuple[float, list]:
    user_msg = build_prompt(task, step, obs, last_reward)
    history.append({"role": "user", "content": user_msg})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                *history,
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Extract just the last line as the ratio after chain-of-thought
        last_line = text.strip().split("\n")[-1].strip()
        ratio = float(last_line)
        ratio = max(0.1, min(0.9, ratio))

        history.append({"role": "assistant", "content": text})
        return ratio, history

    except Exception:
        history.append({"role": "assistant", "content": "0.5"})
        return 0.5, history


def run_task(task: str, client: OpenAI) -> Tuple[bool, int, List[float], float]:
    # No fixed seed — random arrivals so agent decisions actually matter
    env = ClinicSchedulerEnvironment(task=task)
    log_start(task, MODEL_NAME)

    obs = env.reset()
    rewards: List[float] = []
    steps_taken = 0
    history = []
    last_reward = None

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        action_value, history = get_model_action(client, task, step, obs, history, last_reward)
        action = ClinicAction(walk_in_ratio=action_value)

        obs = env.step(action)
        last_reward = obs.reward or 0.0
        rewards.append(last_reward)
        steps_taken = step

        log_step(step, action_value, last_reward, obs.done)

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