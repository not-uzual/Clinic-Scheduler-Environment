from typing import Dict, Any, Literal
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class ClinicAction(Action):
    walk_in_ratio: float = Field(
        ...,
        ge=0.1,
        le=0.9,
        description="Fraction of hourly capacity allocated to walk-ins."
    )


class ClinicObservation(Observation):
    hour: int = Field(..., ge=0, le=8)
    walk_in_queue: int = Field(..., ge=0)
    reserved_queue: int = Field(..., ge=0)
    walk_in_slots: int = Field(..., ge=0, le=10)
    reserved_slots: int = Field(..., ge=0, le=10)
    demand_level: Literal["easy", "medium", "hard"]
    reward: float | None = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class ClinicState(State):
    episode_id: str = ""
    step_count: int = 0
    hour: int = 0

    walk_in_queue: int = 0
    reserved_queue: int = 0

    total_walk_in_arrivals: int = 0
    total_reserved_arrivals: int = 0
    total_served: int = 0
    total_no_shows: int = 0

    cumulative_wait_cost: float = 0.0
    walk_in_slots: int = 5
    reserved_slots: int = 5
    task: Literal["easy", "medium", "hard"] = "medium"