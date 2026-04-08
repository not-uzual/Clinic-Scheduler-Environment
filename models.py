from openenv.core.env_server import Action, Observation, State
from typing import Dict, Any


class ClinicAction(Action):
    walk_in_ratio: float


class ClinicObservation(Observation):
    patients_waiting: float
    walk_in_slots: int
    reserved_slots: int
    hour: int
    reward: float
    done: bool
    info: Dict[str, Any]


class ClinicState(State):
    episode_id: str
    hour: int
    patients_waiting: float
    total_patients: float
    total_wait_time: float
    no_shows: float
    walk_in_slots: int
    reserved_slots: int