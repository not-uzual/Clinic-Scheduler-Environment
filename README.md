---
title: Clinic Scheduler Environment
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
# Clinic Scheduler Environment

An OpenEnv-compatible reinforcement learning environment for clinic appointment scheduling.

The agent learns how to allocate clinic capacity between walk-in patients and reserved appointments in order to reduce wait time, no-shows, and queue buildup across different demand levels.

## Problem Statement

Clinics have limited hourly capacity and must balance:
- Walk-in demand.
- Reserved appointments.
- No-shows.
- Queue buildup during peak hours.

This environment turns that scheduling problem into a compact RL benchmark with three difficulty settings:
- `easy`
- `medium`
- `hard`

The agent chooses a `walk_in_ratio` each step to control how much capacity is reserved for walk-ins versus scheduled patients.

## Why this environment matters

This is a realistic resource-allocation problem with competing goals:
- Keep wait times low.
- Avoid wasted capacity from no-shows.
- Handle demand spikes.
- Maintain stable performance across task difficulty levels.

The environment is designed to be simple enough to train on, but rich enough to expose scheduling tradeoffs.

## Repository Structure

```text
clinic_scheduler/
├── models.py
├── client.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── server/
│   ├── app.py
│   └── clinic_scheduler_environment.py
└── README.md
```

## Environment Design

### Action
- `walk_in_ratio: float`
- Range: `0.1` to `0.9`

### Observation
The environment returns:
- current hour
- walk-in queue
- reserved queue
- walk-in slots
- reserved slots
- task difficulty
- reward
- done flag
- info dictionary with summary metrics

### Reward
The reward is penalty-based and reflects:
- queue backlog
- no-shows
- idle capacity
- allocation mismatch

Negative rewards are expected here because the environment models costs.

## Difficulty Levels

### Easy
Low demand and mild variability. Designed to test basic scheduling behavior.

### Medium
Balanced demand with a peak period. Requires better allocation decisions.

### Hard
Higher baseline demand with stronger peak pressure. This is the most difficult setting.

## How It Works

At each step:
1. The agent chooses a `walk_in_ratio`.
2. The environment splits capacity into walk-in and reserved slots.
3. New clinic demand arrives.
4. Patients are served based on the chosen allocation.
5. The environment returns:
   - the next observation,
   - the step reward,
   - whether the episode is finished,
   - summary metrics in `info`.

## Setup

### 1. Create and activate environment
```bash
python -m venv openenv-clinic-scheduler
source openenv-clinic-scheduler/bin/activate
```

On Windows:
```bash
python -m venv openenv-clinic-scheduler
openenv-clinic-scheduler\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

If you are using `uv`:
```bash
uv sync
```

### 3. Set environment variables
Create a `.env` file or export these in your shell:

```bash
OPENAI_API_KEY=your_api_key_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

## Run the environment server

```bash
python -m server.app
```

The server runs on:
```text
http://localhost:8000
```

## Run the inference script

```bash
python inference.py
```

This runs all three tasks:
- easy
- medium
- hard

It prints:
- step-by-step actions,
- rewards,
- task-level score,
- final summary.

## Expected Output

The script logs each run with:
- `[START]`
- `[STEP]`
- `[END]`

At the end, it prints a task summary with:
- score
- average reward
- steps completed

## Hackathon Evaluation

This submission is designed to be judged on:
- OpenEnv compatibility
- meaningful reward shaping
- difficulty separation
- reproducible execution
- clear clinic-scheduling realism

The benchmark is calibrated so:
- easy should generally score higher than medium,
- medium should generally score higher than hard.

## Notes on Rewards

The rewards are intentionally negative in many steps because the environment models penalties such as:
- wait time,
- backlog,
- idle capacity,
- no-shows.

In RL, negative rewards are normal when the task is framed as cost minimization.

## Demo Flow

A typical demo flow is:

1. Start the server.
2. Run `inference.py`.
3. Show the three task scores.
4. Explain how different `walk_in_ratio` choices affect clinic efficiency.

## Example Use Case

This environment can be used to explore:
- allocation policy learning,
- queue management,
- scheduling under uncertainty,
- healthcare operations optimization.

## Submission Checklist

Before submission, make sure:
- the environment runs locally,
- the inference script completes all three tasks,
- the README explains setup and usage,
- the repo is public,
- the demo is reproducible.
