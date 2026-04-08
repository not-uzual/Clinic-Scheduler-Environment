import uuid
import random
from openenv.core.env_server import Environment
from clinic_scheduler.models import ClinicAction, ClinicObservation, ClinicState


class ClinicSchedulerEnvironment(Environment):
    MAX_TIME = 8
    TOTAL_SLOTS = 10
    
    def __init__(self, task: str = "medium"):
        """Initialize environment with task difficulty.
        
        Args:
            task: "easy" (low demand, simple), "medium" (balanced), "hard" (peak surge)
        """
        super().__init__()
        self.task = task
        self._state = ClinicState(
            episode_id="",
            hour=0,
            patients_waiting=0.0,
            total_patients=0.0,
            total_wait_time=0.0,
            no_shows=0.0,
            walk_in_slots=6,
            reserved_slots=4,
        )

    def reset(self) -> ClinicObservation:
        self._state = ClinicState(
            episode_id=str(uuid.uuid4()),
            hour=0,
            patients_waiting=0.0,
            total_patients=0.0,
            total_wait_time=0.0,
            no_shows=0.0,
            walk_in_slots=6,
            reserved_slots=4,
        )
        return self._to_observation(reward=0.0, done=False)

    def _to_observation(self, reward: float, done: bool) -> ClinicObservation:
        avg_wait = (
            self._state.total_wait_time / (self._state.total_patients + 1e-6)
        )
        return ClinicObservation(
            patients_waiting=self._state.patients_waiting,
            walk_in_slots=self._state.walk_in_slots,
            reserved_slots=self._state.reserved_slots,
            hour=self._state.hour,
            reward=reward,
            done=done,
            info={
                "avg_wait": avg_wait,
                "no_shows": self._state.no_shows,
                "hour": self._state.hour,
                "task": self.task,
            },
        )

    def _get_arrival_range(self) -> tuple:
        """Get arrival range based on task difficulty and hour."""
        is_peak = self._state.hour in [2, 3, 4]
        
        if self.task == "easy":
            # Low demand, no peaks
            return (0.5, 2.0) if not is_peak else (0.5, 2.0)
        elif self.task == "medium":
            # Balanced with peaks
            return (1.0, 4.0) if not is_peak else (4.0, 7.0)
        else:  # hard
            # High demand, aggressive peaks
            return (2.0, 5.0) if not is_peak else (5.5, 9.0)

    def step(self, action: ClinicAction) -> ClinicObservation:
        ratio = max(0.1, min(0.9, action.walk_in_ratio))
        self._state.walk_in_slots = round(self.TOTAL_SLOTS * ratio)
        self._state.reserved_slots = self.TOTAL_SLOTS - self._state.walk_in_slots

        min_arr, max_arr = self._get_arrival_range()
        arrivals = random.uniform(min_arr, max_arr)
        
        self._state.patients_waiting += arrivals
        self._state.total_patients += arrivals

        treated = min(self._state.patients_waiting, 1.0)
        self._state.patients_waiting -= treated

        self._state.total_wait_time += arrivals

        # No-shows: realistic rate (10% of reserved slots, ~0.4-1.0 per hour)
        max_no_show_rate = self._state.reserved_slots * 0.10
        no_shows_this_hour = random.uniform(0.0, max_no_show_rate)
        self._state.no_shows += no_shows_this_hour

        self._state.hour += 1

        avg_wait = self._state.total_wait_time / (self._state.total_patients + 1e-6)
        # Reward: baseline 3.0 minus penalties. Perfect (0 wait, 0 no-shows) = +3.0
        reward = 3.0 - (avg_wait + 1.0 * self._state.no_shows)

        done = self._state.hour >= self.MAX_TIME

        return self._to_observation(reward=reward, done=done)

    @property
    def state(self) -> ClinicState:
        return self._state