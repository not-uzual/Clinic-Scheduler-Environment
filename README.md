---
title: Clinic Scheduler Environment
emoji: 
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

An OpenEnv environment where an intelligent agent optimizes clinic appointment scheduling by deciding the walk-in vs. reserved slot ratio each hour over an 8-hour day. The agent learns to manage peaks, avoid no-shows, and minimize patient wait times.

## Quick Start

```python
from client import ClinicEnv, ClinicAction

# Connect to environment
env = ClinicEnv(base_url="http://localhost:8000")

# Reset environment (start new episode)
result = env.reset()

# Run 8-hour episode
for hour in range(8):
    # Decide walk-in ratio (0.1-0.9) for this hour
    action = ClinicAction(walk_in_ratio=0.6)  # 60% walk-in, 40% reserved
    
    result = env.step(action)
    
    print(f"Hour {hour+1}:")
    print(f"  Patients waiting: {result.observation.patients_waiting}")
    print(f"  Reward: {result.reward:.2f}")
    print(f"  Done: {result.done}")
    
    if result.done:
        break

env.close()
```

## How It Works

### Environment Overview

**Goal**: Manage a clinic's appointment slots optimally across 8 hours to maximize patient satisfaction while minimizing no-shows.

**Challenge**: 
- Only 1 patient/hour treatment capacity (bottleneck!)
- Peak demand surge during hours 3-5 (4-7 arrivals vs. 1-4 normal)
- Walk-in patients arrive unpredictably; reserved patients no-show
- Each decision affects wait times and penalties

## Core Mechanics

### Reset

**What happens**: Initializes a new 8-hour episode.

```python
result = env.reset()
observation = result.observation
```

**Returns** `StepResult` containing:
- `observation` (ClinicObservation): Initial state
- `reward` (float): Always 0.0
- `done` (bool): Always False
- `raw` (dict): Raw API response

**Initial State** (Hour 0):
- 10 total appointment slots (can be split 0.1-0.9 ratio)
- 0 patients waiting
- 0 total wait time accumulated
- 0 no-shows

### State (ClinicState)

Tracks the full episode history:

| Field | Type | Meaning |
|-------|------|---------|
| `episode_id` | str | Unique identifier for this episode |
| `hour` | int | Current hour (0-7) |
| `patients_waiting` | float | Patients in queue waiting for treatment |
| `total_patients` | float | Cumulative arrivals across all hours |
| `total_wait_time` | float | Sum of all patients' wait times |
| `no_shows` | float | Cumulative reserved patients who didn't show |
| `walk_in_slots` | int | Number of walk-in slots THIS hour |
| `reserved_slots` | int | Number of reserved slots THIS hour |

### Action (ClinicAction)

The agent makes **one decision per hour**:

```python
action = ClinicAction(walk_in_ratio=0.6)
```

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `walk_in_ratio` | float | [0.1, 0.9] | % of 10 slots allocated to walk-ins |

**Example**:
- `walk_in_ratio=0.6` -> 6 walk-in slots, 4 reserved slots
- `walk_in_ratio=0.2` -> 2 walk-in slots, 8 reserved slots

**Strategic Decision**:
- **High ratio (0.7-0.9)**: More walk-ins, fewer no-shows, but less predictability
- **Low ratio (0.1-0.3)**: More reserved, more no-shows (COSTLY!), better capacity planning
- **During peak hours (3-5)**: Must increase ratio to handle 4-7 arrivals

### Step

**What happens**: Advance the environment one hour and get feedback.

```python
action = ClinicAction(walk_in_ratio=0.6)
result = env.step(action)
```

**Step Execution**:
1. Clamp ratio to [0.1, 0.9] and calculate slots
   - `walk_in_slots = round(10 * ratio)`
   - `reserved_slots = 10 - walk_in_slots`

2. Generate patient arrivals
   - **Normal hours (1-2, 6-8)**: `random.uniform(1.0, 4.0)` patients
   - **Peak hours (3-5)**: `random.uniform(4.0, 7.0)` patients [PEAK]

3. Add patients to waiting queue
   - `patients_waiting += arrivals`

4. Treat patients (bottleneck!)
   - Only 1 patient can be treated per hour
   - `patients_treated = min(patients_waiting, 1.0)`
   - `patients_waiting -= patients_treated`

5. Accumulate wait time
   - Every arriving patient counts as waiting that hour
   - `total_wait_time += arrivals`

6. Calculate no-shows
   - Reserved patients have probability of not showing
   - `no_shows_this_hour = random.uniform(0, reserved_slots / 2.0)`
   - `no_shows += no_shows_this_hour`

7. Calculate reward (see next section)

8. Increment hour, check if done (hour >= 8)

**Returns** `StepResult`:
- `observation`: Current hour's state + metadata
- `reward`: Performance feedback
- `done`: True if hour >= 8
- `raw`: Raw API response

### Observation (ClinicObservation)

What the agent sees each step:

| Field | Type | Meaning |
|-------|------|---------|
| `patients_waiting` | float | Current queue size |
| `walk_in_slots` | int | Today's walk-in slots (THIS hour) |
| `reserved_slots` | int | Today's reserved slots (THIS hour) |
| `hour` | int | Current hour (0-7) |
| `reward` | float | Reward for THIS step |
| `done` | bool | Is episode finished? |
| `info` | dict | Metadata: avg_wait, no_shows, hour |

**Example observation** (Hour 3, peak demand):
```python
ClinicObservation(
    patients_waiting=3.5,
    walk_in_slots=7,
    reserved_slots=3,
    hour=3,
    reward=1.23,
    done=False,
    info={
        'avg_wait': 2.1,
        'no_shows': 0.8,
        'hour': 3
    }
)
```

## Reward Function

**Formula**:
$$\text{reward} = 3.0 - (\text{avg\_wait} + 1.0 \times \text{no\_shows})$$

Where:
- `avg_wait = total_wait_time / total_patients` (average waiting time per patient)
- `no_shows = cumulative no-show count` (heavily penalized!)

### Reward Breakdown

| Scenario | Calculation | Reward |
|----------|-------------|--------|
| **Perfect** (0 wait, 0 no-shows) | 3.0 - (0 + 0) | **+3.0** [OK] |
| **Good** (1.0 wait, 0.5 no-shows) | 3.0 - (1.0 + 0.5) | **+1.5** [OK] |
| **Okay** (1.5 wait, 1.0 no-shows) | 3.0 - (1.5 + 1.0) | **+0.5** [WARN] |
| **Bad** (2.0 wait, 2.0 no-shows) | 3.0 - (2.0 + 2.0) | **-1.0** [FAIL] |
| **Terrible** (2.5 wait, 3.0 no-shows) | 3.0 - (2.5 + 3.0) | **-2.5** [FAIL] |

### Why Continuous Rewards Matter

- **Discrete rewards** (0, 1, 2...) don't guide learning
- **Continuous rewards** (0.23, 1.78, -0.45...) provide granular feedback
- Small differences in decisions produce different rewards, enabling agent learning

### Reward Components Explained

#### 1. Average Wait Time (`avg_wait`)
- Penalizes queue buildup
- `avg_wait = total_wait_time / total_patients`
- Each new arrival adds to total_wait_time, even if they can't be treated immediately
- **Strategy**: Keep `patients_waiting` low by planning appropriate ratio

#### 2. No-Shows Penalty (1.0x multiplier)
- Realistic penalty for unreliable reservations
- `no_shows_this_hour = random.uniform(0, reserved_slots * 0.10)` (~0.4-1.0/hour)
- More reserved slots -> higher no-show risk
- **Strategy**: Balance reserved slots (predictability) vs walk-ins (reliability)

#### 3. Baseline (3.0)
- Allows both positive and negative rewards
- More realistic given typical no-show rates (~10% of reserved slots)
- Perfect performance (+3.0) requires minimal wait and no-shows
- Good performance (1.0-2.0) is achievable with strategic planning

## Peak Demand Challenge

The environment includes a **realistic stress test**:

**Hours 1-2, 6-8** (Normal):
- 1-4 arrivals/hour
- Manageable with balanced ratio

**Hours 3-5** (PEAK SURGE):
- 4-7 arrivals/hour (2x volume!)
- Forces strategic decisions
- Too low ratio -> massive queues, high wait times
- Too high ratio -> too many walk-ins, can't control

**Graph** (Approximate arrivals):
```
Peak Hour Challenge:
7 |        ^^^
6 |        ^^^
5 |        ^^^
4 | /\    ^^^    /\
3 | /\    ^^^    /\
2 | /\            /\
1 |__________________|
  |_1__2__3__4__5__6__7__8
```

Agent must **anticipate** peak hours and adjust ratios in advance.

## Building & Deployment

### Building the Docker Image

```bash
# From project root
docker build -t clinic_scheduler-env:latest -f server/Dockerfile .
```

### Running Locally

Start the FastAPI server:

```bash
# Option 1: Use server entry point
python -m clinic_scheduler.server.app

# Option 2: Direct uvicorn
uvicorn clinic_scheduler.server.app:app --reload --port 8000
```

Server runs on `http://localhost:8000`

### Deploying to Hugging Face Spaces

Deploy to Hugging Face with:

```bash
# First, create the space manually at https://huggingface.co/spaces/new
# Or let openenv create it (requires write-enabled token)

openenv push --repo-id your-username/clinic-scheduler-env
```

**Prerequisites**:
- HF token with `write` permissions in `.env` file
- Or create space manually first at https://huggingface.co/spaces/new

**After deployment**, access at:
```
https://huggingface.co/spaces/your-username/clinic-scheduler-env
```

## Example: LLM Agent

`inference.py` demonstrates an LLM agent making strategic decisions:

```python
from client import ClinicEnv, ClinicAction
from openai import OpenAI

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key="hf_xxx")
env = ClinicEnv(base_url="http://localhost:8000")

result = env.reset()

for hour in range(8):
    # Ask LLM to decide ratio based on state
    prompt = f"""
    Hour {result.observation.hour + 1}:
    - Patients waiting: {result.observation.patients_waiting}
    - Upcoming: PEAK SURGE starting hour 3!
    
    Choose walk_in_ratio (0.1-0.9). Only output the number.
    """
    
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    
    ratio = float(response.choices[0].message.content.strip())
    action = ClinicAction(walk_in_ratio=ratio)
    
    result = env.step(action)
    print(f"Hour {hour+1}: ratio={ratio:.2f} -> reward={result.reward:.2f}")
    
    if result.done:
        break

env.close()
```

**Key Points**:
- Agent sees real-time state (patients waiting, hour)
- System prompt warns about peak hours
- Agent makes 8 decisions (one per hour)
- Rewards guide learning over multiple episodes

## Project Structure

```
clinic_scheduler/
 __init__.py                    # Module exports
 .env                           # API keys & config (git-ignored)
 .gitignore                     # Git exclusions
 README.md                      # This documentation
 openenv.yaml                   # OpenEnv manifest (spec_version, name, port)
 pyproject.toml                 # Dependencies: openenv-core, openai, requests, python-dotenv
 client.py                      # HTTP client: ClinicEnv class
 models.py                      # Pydantic models: ClinicAction, ClinicObservation, ClinicState
 inference.py                   # LLM agent script (runs episodes with Qwen2.5)

 server/
    __init__.py
    app.py                     # FastAPI server entry point
    clinic_scheduler_environment.py  # Core simulation logic
    Dockerfile                 # Docker build recipe

 openenv_clinic_scheduler.egg-info/
     (auto-generated by setuptools)
```

### Key Files Explained

#### `models.py` - Type Definitions
Defines the contract between agent and environment:
- **ClinicAction**: Agent decides `walk_in_ratio` (float)
- **ClinicObservation**: State agent sees each step
- **ClinicState**: Full episode state (tracked server-side)

#### `client.py` - HTTP Client Wrapper
Synchronous client for agents:
- `ClinicEnv(base_url)` - Connect to server
- `env.reset()` - Start new episode
- `env.step(action)` - Advance one hour
- `env.close()` - Cleanup

#### `server/clinic_scheduler_environment.py` - Simulation Engine
Core environment logic:
- `reset()` - Initialize episode, zero all counters
- `step(action)` - Execute one hour of clinic operations
- State tracking - Maintains `patients_waiting`, `total_wait_time`, `no_shows`
- Reward calculation - Applies penalty formula
- Peak demand logic - Surge in hours 3-5

#### `server/app.py` - FastAPI Server
HTTP API endpoints:
- `/` - Health check
- `/reset` - `POST` -> `ClinicObservation`
- `/step` - `POST` with action -> `ClinicObservation` + reward

#### `inference.py` - Multi-Task Graded Evaluation 
OpenEnv compliance implementation:
- Runs **3 graded tasks** (easy, medium, hard)
- **Deterministic graders** scoring 0.0 < score < 1.0
- Task-specific system prompts and difficulty levels
- Programmatic success/failure criteria
- Full [START]/[STEP]/[END] output format with grader scores

---

## Understanding Reward Flow (Complete Example)

Let's trace one complete step:

**Initial state (Hour 0)**:
```python
result = env.reset()
# observation.hour = 0
# observation.patients_waiting = 0.0
# observation.walk_in_slots = 6 (default)
# observation.reserved_slots = 4 (default)
# result.reward = 0.0
# System prompt warns about peak hours 3-5
```

**Hour 1 - Agent decides**:
```python
action = ClinicAction(walk_in_ratio=0.4)
# -> 4 walk-in slots, 6 reserved slots
```

**Hour 1 - Simulation**:
1. Generate arrivals: `random.uniform(1.0, 4.0)` -> says 2.5 patients arrive
2. Add to queue: `patients_waiting = 0.0 + 2.5 = 2.5`
3. Treat patients: `min(2.5, 1.0)` -> 1 treated, 1.5 still waiting
4. Update wait time: `total_wait_time = 0.0 + 2.5 = 2.5`
5. No-shows: `random.uniform(0, 6/2)` -> 1.8 patients no-show (reserved slots high!)
6. Calculate reward:4/10)` -> 0.3 patients no-show (realistic rate)
6. Calculate reward:
   - `avg_wait = 2.5 / 2.5 = 1.0`
   - `no_shows = 0.3`
   - `reward = 3.0 - (1.0 + 0.3) = 1.7` [OK] (good performance)
**Hour 1 - Agent receives**:
```python
result = env.step(action)
# observation.hour = 1
# observation.patients_waiting = 1.5  (still in queue)
# observation.walk_in_slots = 4
# observation.reserved_slots = 6
# observation.reward = -0.8
# observation.done = Fa1.7
# observation.done = False
# observation.info = {'avg_wait': 1.0, 'no_shows': 0.3

**Agent learns**: 
- Too many reserved slots (0.4 ratio) -> high no-shows (1.8) -> reward penalized
- Next hour should increase ratio to reduce reserved slots

---

## Tips for Agents

### Adaptive Strategy
1. **Hours 1-2**: Conservative (0.3-0.4 ratio) - demand is low
2. **Hours 3-5**: Aggressive (0.7-0.9 ratio) - PEAK SURGE requires walk-ins
3. **Hours 6-8**: Moderate (0.5-0.6 ratio) - demand drops, clear backlog

### Reward Interpretation
- **Reward > 1.5**: Excellent! Low wait, minimal no-shows
- **Reward 0.2.0**: Excellent! Low wait, minimal no-shows
- **Reward 1.0-2.0**: Good management
- **Reward 0-1.0**: Okay, some issues
- **Reward < 0
### Key Insight
**No-shows are the biggest penalty** - they hurt twice:
1. Cost capacity (1.0x multiplier directly)
2. Can't be predicted, forcing conservative ratios
3. Conservative ratios cause queues (avg_wait increases)

Minimize no-shows by increasing walk-in ratio!

---

## OpenEnv Compliance: Graded Tasks

The environment includes **3 graded tasks** (easy -> medium -> hard) with programmatic graders that score agent performance **strictly between 0.0 and 1.0**.

### Task Definitions

| Task | Difficulty | Description | Baseline Arrivals | Peak Surge | Success Threshold |
|------|-----------|-------------|-------------------|-----------|-------------------|
| **easy** |  Easy | Low demand, no peaks | 0.5-2.0/hour | None | > 0.65 |
| **medium** |  Medium | Balanced with peaks | 1-4/hour | 4-7/hour (hours 3-5) | > 0.50 |
| **hard** |  Hard | High demand, aggressive surge | 2-5/hour | 5.5-9/hour (hours 3-5) | > 0.35 |

### Grader Scoring

Each task has a **deterministic grader** that evaluates:
- **Average reward** (primary metric)
- **Episode completion** (bonus points)
- **Reward trajectory** (consistency)

**Scoring Formula** (task-dependent):
- Easy: `0.1 + 0.8 * normalize(avg_reward)` -> range (0.05, 0.95)
- Medium: `0.15 + 0.7 * normalize(avg_reward)` -> range (0.1, 0.9)
- Hard: `0.1 + 0.8 * normalize(avg_reward)` -> range (0.05, 0.95)

**Completion Bonus**: +0.02 for finishing all 8 steps

**Final Score**: Strictly in range **[0.01, 0.99]** 

### Running Graded Tasks

```bash
# Run all 3 graded tasks
python inference.py

# Output format:
# [START] task=easy env=clinic_scheduler model=Qwen/Qwen2.5-72B-Instruct
# [STEP] step=1 action=walk_in_ratio=0.45 reward=1.23 done=false error=null
# [STEP] step=2 action=walk_in_ratio=0.52 reward=0.87 done=false error=null
# ...
# [END] task=easy success=true steps=8 rewards=[1.23,0.87,...] grader_score=0.7234
#
# TASK SUMMARY
# EASY     PASS  score=0.7234 avg_reward=+1.05 steps=8/8
# MEDIUM   PASS  score=0.6145 avg_reward=+0.73 steps=8/8
# HARD      PASS  score=0.4567 avg_reward=-0.12 steps=8/8
# ==============================================================
# Average Score: 0.5982 (range 0.0-1.0)
```
## Configuration

Edit `.env` for:
- `OPENAI_API_KEY`: OpenAI or Hugging Face token (required for inference)
- `API_KEY`: Alternative name for the token (backward compatible)
- `API_BASE_URL`: LLM endpoint (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME`: Model to use (default: `Qwen/Qwen2.5-72B-Instruct`)
- `LOCAL_IMAGE_NAME`: Docker image name (default: `clinic_scheduler-env:latest`)
---

## Support & Questions

For issues or features:
1. Check `openenv.yaml` for environment metadata
2. Review `server/clinic_scheduler_environment.py` for simulation details
3. Run `python inference.py` to test LLM integration
4. Check server logs: `python -m clinic_scheduler.server.app`
