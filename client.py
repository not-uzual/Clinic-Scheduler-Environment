import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from models import ClinicAction, ClinicObservation, ClinicState


@dataclass
class StepResult:
    observation: ClinicObservation
    reward: float
    done: bool
    raw: Dict[str, Any]


class ClinicEnv:
    """HTTP client for Clinic Scheduler environment"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    @staticmethod
    def from_docker_image(image_name: str) -> "ClinicEnv":
        """Create environment from Docker image.
        
        For now, this just creates a client assuming the server is already running.
        In a full implementation, this would start the Docker container.
        """
        return ClinicEnv(base_url="http://localhost:8000")
    
    def reset(self) -> StepResult:
        """Reset the environment"""
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        data = response.json()
        
        obs_data = data.get("observation", data)
        # Add default reward and done if not present
        if "reward" not in obs_data:
            obs_data["reward"] = 0.0
        if "done" not in obs_data:
            obs_data["done"] = False
            
        obs = ClinicObservation(**obs_data)
        
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            raw=data
        )
    
    def step(self, action: ClinicAction) -> StepResult:
        """Take a step in the environment"""
        payload = {
            "action": {
                "walk_in_ratio": action.walk_in_ratio,
            }
        }
        
        response = requests.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        data = response.json()
        
        obs_data = data.get("observation", data)
        # Add reward and done from top level if not in observation
        if "reward" not in obs_data:
            obs_data["reward"] = data.get("reward", 0.0)
        if "done" not in obs_data:
            obs_data["done"] = data.get("done", False)
            
        obs = ClinicObservation(**obs_data)
        
        return StepResult(
            observation=obs,
            reward=data.get("reward", obs_data.get("reward", 0.0)),
            done=data.get("done", obs_data.get("done", False)),
            raw=data
        )
    
    def close(self) -> None:
        """Close the environment and cleanup resources"""
        pass
