import random
import uuid
from typing import Optional

from openenv.core.env_server import Environment
from clinic_scheduler.models import ClinicAction, ClinicObservation, ClinicState


class ClinicSchedulerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    MAX_TIME = 8
    TOTAL_SLOTS = 10

    # Reward normalization constants
    MAX_POSSIBLE_PENALTY = (
        0.42 * 10 +  # max reserved backlog
        0.28 * 10 +  # max walk-in backlog
        0.60 * 6  +  # max no-shows (hard mode peak)
        0.25 * 10 +  # max idle slots
        0.18 * 1     # max allocation penalty
    )  # ≈ 11.44
    MAX_POSSIBLE_BONUS = 0.18 * 10  # = 1.8
    NORM_FACTOR = MAX_POSSIBLE_PENALTY  # normalize by worst case

    def __init__(self, task: str = "medium", seed: Optional[int] = None):
        super().__init__()
        self.task = task if task in {"easy", "medium", "hard"} else "medium"
        self.rng = random.Random(seed)
        self._state = ClinicState(task=self.task)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ClinicObservation:
        if seed is not None:
            self.rng.seed(seed)

        self._state = ClinicState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            hour=0,
            walk_in_queue=0,
            reserved_queue=0,
            total_walk_in_arrivals=0,
            total_reserved_arrivals=0,
            total_served=0,
            total_no_shows=0,
            cumulative_wait_cost=0.0,
            walk_in_slots=5,
            reserved_slots=5,
            task=self.task,
        )
        return self._to_observation(reward=0.0, done=False)

    def _arrival_profile(self) -> tuple[int, int, int, int]:
        peak = self._state.hour in {2, 3, 4}

        if self.task == "easy":
            return (0, 2, 1, 2) if not peak else (0, 2, 1, 2)
        if self.task == "medium":
            return (1, 3, 1, 3) if not peak else (3, 5, 2, 4)
        return (2, 4, 2, 4) if not peak else (4, 6, 3, 5)

    def _no_show_rate(self) -> float:
        if self.task == "easy":
            return 0.08
        if self.task == "medium":
            return 0.12
        return 0.18

    def _to_observation(self, reward: float | None, done: bool) -> ClinicObservation:
        return ClinicObservation(
            hour=self._state.hour,
            walk_in_queue=self._state.walk_in_queue,
            reserved_queue=self._state.reserved_queue,
            walk_in_slots=self._state.walk_in_slots,
            reserved_slots=self._state.reserved_slots,
            demand_level=self.task,
            reward=reward,
            done=done,
            info={
                "total_served": self._state.total_served,
                "total_no_shows": self._state.total_no_shows,
                "cumulative_wait_cost": round(self._state.cumulative_wait_cost, 2),
                "step_count": self._state.step_count,
            },
        )

    def step(self, action: ClinicAction, timeout_s=None, **kwargs) -> ClinicObservation:
        s = self._state

        ratio = max(0.1, min(0.9, action.walk_in_ratio))
        s.walk_in_slots = int(round(self.TOTAL_SLOTS * ratio))
        s.walk_in_slots = min(max(s.walk_in_slots, 1), 9)
        s.reserved_slots = self.TOTAL_SLOTS - s.walk_in_slots

        w_min, w_max, r_min, r_max = self._arrival_profile()
        new_walk_ins = self.rng.randint(w_min, w_max)
        new_reserved = self.rng.randint(r_min, r_max)

        s.walk_in_queue += new_walk_ins
        s.reserved_queue += new_reserved
        s.total_walk_in_arrivals += new_walk_ins
        s.total_reserved_arrivals += new_reserved

        reserved_candidates = min(s.reserved_queue, s.reserved_slots)
        no_shows = sum(
            1 for _ in range(reserved_candidates)
            if self.rng.random() < self._no_show_rate()
        )

        reserved_served = max(0, reserved_candidates - no_shows)
        walk_in_served = min(s.walk_in_queue, s.walk_in_slots)

        reserved_idle = max(0, s.reserved_slots - reserved_candidates)
        walk_in_idle = max(0, s.walk_in_slots - min(s.walk_in_queue, s.walk_in_slots))
        idle_slots = reserved_idle + walk_in_idle

        s.reserved_queue -= reserved_served
        s.walk_in_queue -= walk_in_served
        s.total_served += reserved_served + walk_in_served
        s.total_no_shows += no_shows

        reserved_backlog = s.reserved_queue
        walk_in_backlog = s.walk_in_queue

        service_bonus = 0.18 * (reserved_served + walk_in_served)
        queue_penalty = 0.42 * reserved_backlog + 0.28 * walk_in_backlog
        no_show_penalty = 0.60 * no_shows
        idle_penalty = 0.25 * idle_slots

        demand_gap = abs(
            (new_walk_ins / max(1, new_walk_ins + new_reserved)) - ratio
        )
        allocation_penalty = 0.18 * demand_gap

        raw_reward = (
            service_bonus
            - queue_penalty
            - no_show_penalty
            - idle_penalty
            - allocation_penalty
        )

        # Normalize to [-1, +1]
        reward = raw_reward / self.NORM_FACTOR
        reward = max(-1.0, min(1.0, reward))

        s.cumulative_wait_cost += reserved_backlog + walk_in_backlog
        s.step_count += 1
        s.hour += 1

        done = s.hour >= self.MAX_TIME

        return self._to_observation(
            reward=round(reward, 3),
            done=done,
        )

    @property
    def state(self) -> ClinicState:
        return self._state