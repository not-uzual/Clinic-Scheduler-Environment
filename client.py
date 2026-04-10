from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from models import ClinicAction, ClinicObservation


@dataclass
class StepResult:
    observation: ClinicObservation
    reward: float
    done: bool
    raw: Dict[str, Any]


class ClinicEnv:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: tuple[float, float] = (3.0, 15.0)):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @staticmethod
    def from_docker_image(image_name: str, base_url: str = "http://localhost:8000") -> "ClinicEnv":
        return ClinicEnv(base_url=base_url)

    def health(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            return response.ok
        except requests.RequestException:
            return False

    def reset(self) -> StepResult:
        try:
            response = self.session.post(f"{self.base_url}/reset", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to reset ClinicEnv at {self.base_url}: {exc}") from exc

        obs_data = data.get("observation", data)
        obs = ClinicObservation(**obs_data)

        return StepResult(
            observation=obs,
            reward=obs.reward or 0.0,
            done=obs.done,
            raw=data,
        )

    def step(self, action: ClinicAction) -> StepResult:
        payload = {"action": action.model_dump() if hasattr(action, "model_dump") else {"walk_in_ratio": action.walk_in_ratio}}

        try:
            response = self.session.post(f"{self.base_url}/step", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to step ClinicEnv at {self.base_url}: {exc}") from exc

        obs_data = data.get("observation", data)
        if "reward" not in obs_data:
            obs_data["reward"] = data.get("reward")
        if "done" not in obs_data:
            obs_data["done"] = data.get("done", False)

        obs = ClinicObservation(**obs_data)

        return StepResult(
            observation=obs,
            reward=obs.reward or 0.0,
            done=obs.done,
            raw=data,
        )

    def close(self) -> None:
        self.session.close()