---
title: Clinic Scheduler Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - scheduling
---

# Clinic Scheduler Environment

An OpenEnv environment where an agent optimizes clinic appointment scheduling by deciding the walk-in vs. reserved slot ratio each hour over an 8-hour day.

## Quick Start

```python
from client import ClinicEnv, ClinicAction

# Connect to environment
env = ClinicEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()

# Run 8-hour episode
for hour in range(8):
    action = ClinicAction(walk_in_ratio=0.6)
    result = env.step(action)
    
    print(f"Hour {hour+1}: Reward: {result.reward:.2f}")
    
    if result.done:
        break

env.close()
```

## Environment Overview

**Goal**: Manage clinic appointments to maximize patient satisfaction and minimize no-shows.

**Challenge**: 
- Only 1 patient/hour treatment capacity
- Peak demand during hours 3-5 (4-7 arrivals)
- Walk-in patients arrive unpredictably; reserved patients may not show
- Balance walk-in reliability vs reserved predictability

## Core Mechanics

### Reset

Initializes a new 8-hour episode. Returns `StepResult` with initial observation.

```python
result = env.reset()
observation = result.observation
```

### State

Tracks episode progress:

| Field | Meaning |
|-------|---------|
| `hour` | Current hour (0-7) |
| `patients_waiting` | Patients in queue |
| `total_wait_time` | Sum of all wait times |
| `no_shows` | Cumulative no-shows |
| `walk_in_slots` | Walk-in slots this hour |
| `reserved_slots` | Reserved slots this hour |

### Action

One decision per hour:

```python
action = ClinicAction(walk_in_ratio=0.6)  # 60% walk-in, 40% reserved
```

- `walk_in_ratio`: Float between 0.1 and 0.9
- 10 total slots per hour split by ratio

### Step

Advances one hour and returns feedback.

```python
result = env.step(action)
```

**Step process**:
1. Calculate walk-in and reserved slots from ratio
2. Generate patient arrivals (1-4 normal hours, 4-7 peak hours)
3. Add patients to waiting queue
4. Treat 1 patient (bottleneck)
5. Calculate no-shows on reserved slots (10% rate)
6. Compute reward based on wait time and no-shows
7. Check if done (hour >= 8)

Returns `StepResult` with observation, reward, done status.

## Reward Function

**Formula**: reward = 3.0 - (avg_wait + no_shows)

Where:
- `avg_wait` = total_wait_time / total_patients
- `no_shows` = cumulative no-show count

### Reward Examples

| Scenario | Calculation | Reward |
|----------|-------------|--------|
| Perfect (0 wait, 0 no-shows) | 3.0 - (0 + 0) | +3.0 |
| Good (1.0 wait, 0.5 no-shows) | 3.0 - (1.0 + 0.5) | +1.5 |
| Okay (1.5 wait, 1.0 no-shows) | 3.0 - (1.5 + 1.0) | +0.5 |
| Bad (2.0 wait, 2.0 no-shows) | 3.0 - (2.0 + 2.0) | -1.0 |

Continuous rewards enable granular agent learning.

## Task Definitions

### Easy Task

- Normal demand: 0.5-2.0 arrivals/hour
- Goal: Achieve grader score > 0.50
- Expected reward: +0.8 to +2.0
- Strategy: Conservative scheduling with balanced ratio

### Medium Task

- Mixed demand: 1-4 arrivals, 4-7 peak hours
- Goal: Achieve grader score > 0.48
- Expected reward: +0.5 to +1.5
- Strategy: Adaptive ratios, higher reserves during normal hours

### Hard Task

- High demand: 2-5 arrivals, 5.5-9 peak hours
- Goal: Achieve grader score > 0.35
- Expected reward: -0.5 to +1.0
- Strategy: Dynamic walk-in adjustment, minimal no-shows

## Deployment

### Local Testing

Build and run the environment:

```bash
openenv build
openenv serve
```

Access at `http://localhost:8000`

### Hugging Face Spaces

Push to HF:

```bash
openenv push --repo-id your-username/clinic-scheduler-env
```

Environment will be live on Hugging Face Spaces.

## API Reference

### ClinicEnv

```python
from client import ClinicEnv

env = ClinicEnv(base_url="http://localhost:8000")
result = env.reset()
result = env.step(action)
env.close()
```

### ClinicAction

```python
from models import ClinicAction

action = ClinicAction(walk_in_ratio=0.6)
```

### ClinicObservation

```python
obs = result.observation
params = {
    'patients_waiting': obs.patients_waiting,
    'walk_in_slots': obs.walk_in_slots,
    'reserved_slots': obs.reserved_slots,
    'hour': obs.hour,
    'done': obs.done
}
```

## Implementation Details

### Framework

- OpenEnv specification compliant
- FastAPI server on port 8000
- Python 3.11+
- Uvicorn ASGI runner

### Key Files

- `server/app.py` - FastAPI application
- `server/clinic_scheduler_environment.py` - Environment logic
- `client.py` - HTTP client
- `models.py` - Type definitions
- `inference.py` - Task evaluation script

### Docker

Built with Docker for easy deployment. Runs FastAPI server on 0.0.0.0:8000.

## License

MIT
