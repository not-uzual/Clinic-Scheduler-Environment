from openenv.core.env_server import create_fastapi_app

from clinic_scheduler.models import ClinicAction, ClinicObservation
from clinic_scheduler.server.clinic_scheduler_environment import ClinicSchedulerEnvironment


def create_environment() -> ClinicSchedulerEnvironment:
    return ClinicSchedulerEnvironment(task="medium", seed=42)


app = create_fastapi_app(
    create_environment,
    ClinicAction,
    ClinicObservation,
)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "clinic_scheduler.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()