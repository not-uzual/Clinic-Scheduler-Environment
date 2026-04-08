import uuid
import random
from openenv.core.env_server import Environment
from clinic_scheduler.models import ClinicAction, ClinicObservation, ClinicState


class ClinicSchedulerEnvironment(Environment):
    MAX_TIME = 8
    TOTAL_SLOTS = 10
    def __init__(self):
        super().__init__()
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
            },
        )

    def step(self, action: ClinicAction) -> ClinicObservation:
        ratio = max(0.1, min(0.9, action.walk_in_ratio))
        self._state.walk_in_slots = round(self.TOTAL_SLOTS * ratio)
        self._state.reserved_slots = self.TOTAL_SLOTS - self._state.walk_in_slots

        # PEAK DEMAND HOURS: 3-5 have surge demand (challenge for LLM!)
        if self._state.hour in [2, 3, 4]:
            arrivals = random.uniform(4.0, 7.0)
        else:
            arrivals = random.uniform(1.0, 4.0)
        
        self._state.patients_waiting += arrivals
        self._state.total_patients += arrivals

        treated = min(self._state.patients_waiting, 1.0)
        self._state.patients_waiting -= treated

        self._state.total_wait_time += arrivals

        # No-shows scale with reserved slots (higher reserved → higher no-show risk)
        no_shows_this_hour = random.uniform(0.0, self._state.reserved_slots / 2.0)
        self._state.no_shows += no_shows_this_hour

        self._state.hour += 1

        avg_wait = self._state.total_wait_time / (self._state.total_patients + 1e-6)
        # Reward: baseline 2.0 minus penalties. Perfect (0 wait, 0 no-shows) = +2.0
        reward = 2.0 - (avg_wait + 1.0 * self._state.no_shows)

        done = self._state.hour >= self.MAX_TIME

        return self._to_observation(reward=reward, done=done)

    @property
    def state(self) -> ClinicState:
        return self._state