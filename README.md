# Pacman RLDP

Pacman RLDP is a team project workspace for dynamic programming (DP) and reinforcement learning experiments on a Pacman runtime.

## Problem Definition
- Task: control Pacman to collect all food while avoiding ghosts.
- Agent objective: maximize cumulative episodic return.
- Environment dynamics: turn-based sequence where Pacman acts first, then each ghost acts.
- Transition type: **stochastic by default** because ghost actions are sampled uniformly from legal actions.

## Environment Specification
Primary API: `pacman_rldp.env.PacmanEnv` (Gymnasium-style).

### State Representation (`ObsDict`)
`reset()` and `step()` return a structured observation dictionary.
Default (`env.observation.name: raw`) fields:
- `pacman_position`: `(2,)` float32
- `ghost_positions`: `(num_ghosts, 2)` float32 (`-1` padding for absent slots)
- `ghost_timers`: `(num_ghosts,)` int32
- `ghost_present`: `(num_ghosts,)` binary
- `walls`: `(width, height)` int8 binary
- `food`: `(width, height)` int8 binary
- `capsules`: `(width, height)` int8 binary
- `score`: `(1,)` float32
- `step_count`: `(1,)` int32

### Observation Registry
Observation format is configurable through `env.observation.name`:
- `raw` (default): full grids (`walls/food/capsules`) + agent features.
- `chunked_food`: chunk-level binary food map + full local maps for Pacman's current chunk.
- `food_bitmask`: one integer food bitmask over walkable (non-wall) cells.
- `bitmask_distance_buckets`: `food_bitmask` + bucketized nearest-food/nearest-ghost distances and coarse directions.

Config block (in `configs/default.yaml`):
```yaml
env:
  observation:
    name: raw
    chunk_w: 4
    chunk_h: 2
    distance_bucket_size: 2
```

Bitmask encoding rule (deterministic):
- only non-wall cells are encoded,
- canonical order is row-major by map rows top-to-bottom, then left-to-right within each row.

### Action Space
- `action_space = Discrete(5)`
- Mapping:
  - `0 -> North`
  - `1 -> South`
  - `2 -> East`
  - `3 -> West`
  - `4 -> Stop`

### Transition Logic
For one `step(action)`:
1. Pacman action is applied (or rejected according to `invalid_action_mode`).
2. Ghosts respond sequentially.
3. Episode ends on win/loss (`terminated=True`) or max-step limit (`truncated=True`).

### Ghost Strategy (Default)
- Config key: `env.ghost_policy`.
- Current supported default: `random`.
- `random` policy means each ghost samples uniformly from its legal actions at its turn.
- Ghost movement rules are as follows:
  - ghost cannot choose `Stop` (except edge-case when no alternatives),
  - ghost usually avoids immediate reverse direction unless forced by map topology.
- Result: transition dynamics are stochastic and reproducible under fixed seed.

### Ghost Strategy (`markovian`)
- Set `env.ghost_policy: markovian` for uniform one-step Markov transitions.
- At each ghost turn:
  - all legal non-`Stop` neighbor transitions are sampled with equal probability,
  - `Stop` transition has probability `0` (unless no move is available).
- This policy is stochastic and reproducible under fixed seed.

### Ghost Strategy (`loop_path`)
- Set `env.ghost_policy: loop_path` for deterministic ghost patrol.
- Additional config:
  - `ghost_loop_matrix`: `0/1` matrix (row-major, top-to-bottom) with `1` for loop cells.
  - `ghost_loop_direction`: currently `clockwise`.
- Validation rules:
  - matrix shape must match layout size,
  - values must be only `0` or `1`,
  - all `1` cells must be non-wall cells,
  - `1` cells must form one closed simple cycle (degree 2 at each path node).
- Multi-ghost behavior:
  - all ghosts use the same single loop,
  - initial loop indices are evenly spaced along that loop.
- If a ghost spawn is off-loop, it is snapped to nearest loop node for loop indexing.
- Dynamics are deterministic (independent of RNG) once loop policy is active.

### What The Agent Observes
- The environment is **fully observable** in the provided observation dictionary.
- Pacman directly receives:
  - exact Pacman position,
  - exact ghost positions and scared timers,
  - full wall grid,
  - full food grid,
  - full capsule map,
  - current score and step count.
- No hidden-state belief model is required for baseline DP experiments.
- Auxiliary `info` returned by `step()` includes legal action IDs and episode flags (`is_win`, `is_lose`).

### Episode End Conditions
- **Termination**:
  - Win: all food consumed.
  - Loss: Pacman collides with a non-scared ghost.
- **Truncation**:
  - `step_count >= max_steps`.

### Reward Function (Default)
Configured in `configs/default.yaml` under `env.reward`.

| Event | Reward |
|---|---:|
| Time step penalty | `-1.0` |
| Food eaten | `+10.0` |
| Capsule eaten | `+0.0` |
| Ghost eaten | `+200.0` |
| Win | `+500.0` |
| Lose | `-500.0` |
| Invalid action | `-5.0` |

## Reproducibility
## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Train (scaffold)
```bash
python scripts/train.py --config configs/default.yaml
```
Expected outputs in `results/train/`:
- `model.pkl` (placeholder artifact for DP model integration)
- `train_metrics.json`
- `episode_returns.csv`
- `learning_curve.png`

## Evaluate (scaffold)
```bash
python scripts/eval.py --config configs/default.yaml --model results/train/model.pkl
```
Expected outputs in `results/eval/`:
- `eval_metrics.json`

Baseline heuristic policy evaluation (`nearest food`, ghost-avoidance for Manhattan `<= 2`):
```bash
python scripts/eval.py --config configs/default.yaml --policy baseline
```
Baseline protocol defaults:
- base seed `42`
- `200` episodes
- per-episode seed schedule: `42 + i`

Additional metrics in `eval_metrics.json`:
- `avg_reward`
- `avg_episode_length`
- `policy`
- `base_seed`

## Manual Play
Autonomous agent (random baseline) with graphics:
```bash
python scripts/play.py --config configs/default.yaml --render-mode human
```

Heuristic baseline agent (nearest food + ghost avoidance):
```bash
python scripts/play.py --config configs/default.yaml --render-mode human --policy baseline
```

Human keyboard mode (Tk):
```bash
python scripts/play.py --config configs/default.yaml --manual --render-mode human
```
Terminal mode (keyboard prompt):
```bash
python scripts/play.py --config configs/default.yaml --manual --render-mode ansi
```

## GIF Export
`scripts/play.py` saves GIF only from real human visualization (`--render-mode human`).
Output directory: `results/important/`.
- Custom filename:
```bash
python scripts/play.py --config configs/default.yaml --render-mode human --gif-title my_run
```
- Disable GIF export:
```bash
python scripts/play.py --config configs/default.yaml --render-mode human --no-gif
```
- Default filename (if `--gif-title` is omitted): `experiment_{local_time}.gif`
- `--render-mode ansi` does not export GIF.

## DP Scaffold Interfaces
- `pacman_rldp.algorithms.MDPModel`
- `pacman_rldp.algorithms.TransitionOutcome`
- `pacman_rldp.algorithms.PacmanMDPAdapter`

These provide model-first abstractions (state encoding, legal actions, transition outcomes, reward, terminal checks) for teammate DP solver implementations.

## Project Structure
- `src/pacman_rldp/third_party/bk/`: environment core, rendering, keyboard control, layouts.
- `src/pacman_rldp/env/`: Gymnasium environment wrapper.
- `src/pacman_rldp/algorithms/`: DP model interfaces and adapter.
- `src/pacman_rldp/agents/`: baseline random/manual policies.
- `scripts/`: train/eval/play entrypoints.
- `configs/`: reproducible experiment configuration.

## Scope Note
This branch implements the team-role deliverables: environment + visualization + manual mode + DP scaffold. Final learning-agent performance claims and baseline outperformance are expected in subsequent team iterations.
