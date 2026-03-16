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

### Observation State Space (Approximate)
Approximate orders below use current default assumptions: `smallClassic`, width `20`, height `7`, walkable cells `64`, ghosts `2`, chunk config `4x2`, max scared timer `40`, max steps `500`.

| Observation | Approximate State Space | Formula (order-level) |
|---|---:|---|
| `raw` | `~1.63e31` | `N_pac * N_ghost^G * 2^N_food * 2^N_caps * (T+1)^G * (S+1)` |
| `chunked_food` | `~3.03e23` | `N_pac * N_ghost^G * (T+1)^G * (S+1) * 2^C * C * 2^(chunk_w*chunk_h) * 2^(chunk_w*chunk_h)` |
| `food_bitmask` | `~4.07e30` | `N_pac * N_ghost^G * 2^N_food * (T+1)^G * (S+1)` |
| `bitmask_distance_buckets` | `~2.61e34` | `food_bitmask_space * N_food_bucket * N_ghost_bucket * N_food_dir * N_ghost_dir` |

Where:
- `N_pac = 64`, `N_ghost = 64`, `G = 2`, `N_food = 64`, `N_caps = 2`,
- `T = 40`, `S = 500`, `C = ceil(20/4)*ceil(7/2)=20`,
- `N_food_bucket = 16`, `N_ghost_bucket = 16`, `N_food_dir = 5`, `N_ghost_dir = 5`.
- These are approximate upper-order estimates and change with layout/config.

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
  - `ghost_loop_direction`: `clockwise` or `anticlockwise`.
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

## Results & Artifacts by Algorithm

### Baseline
- Loop-path demo: `results/important/loop2_baseline.gif`
- Markovian demo: `results/important/markov1_baseline.gif`
- Random-policy game demo (`game1 == random policy`): `results/important/game1_baseline.gif`

### Q-learning
- Rollout video: `results/important/Q_learning.mp4`
- Training curve: `results/Q_learning_training_curve.jpg`

### Value Iteration (food-bitmask empirical VI)
- Fast/high-return rollout: `results/important/fast_high_return_VI.gif`
- Best overall rollout (requested `best_overall.gif`, repository file is): `results/important/best_overall_VI.gif`
- Training reward curve: `results/important/train_bitmask_vi_reward_curve.png`

### SARSA (Future Placeholder)
- Qualitative artifact (gif/mp4): `TBD`
- Training curve: `TBD`
- Metrics summary: `TBD`

### Policy Iteration (Future Placeholder)
- Qualitative artifact (gif/mp4): `TBD`
- Training curve: `TBD`
- Metrics summary: `TBD`

## Mathematical Formulas Used in Code

### Baseline Heuristic (`src/pacman_rldp/agents/baseline.py`)
- Danger check with Manhattan distance:
  - `d_ghost(s) = min_g ||p_pac - p_g||_1`
  - Escape mode if `d_ghost(s) <= d_threshold`
- Escape action selection:
  - `a* = argmax_{a in A_legal} min_g ||p_pac'(a) - p_g||_1`
- Food chasing (when safe):
  - choose first action on BFS shortest path to nearest food.

### Q-learning (`scripts/q_learning_agent.py`)
- Linear approximation:
  - `Q(s,a) = w^T f(s,a)`
- TD target:
  - terminal: `y = r`
  - non-terminal: `y = r + gamma * max_{a'} Q(s',a')`
- TD error:
  - `delta = y - Q(s,a)`
- Weight update:
  - `w_i <- w_i + alpha * delta * f_i(s,a)`
- Action selection:
  - epsilon-greedy over legal actions.

### Value Iteration (`src/pacman_rldp/algorithms/food_bitmask_value_iteration.py`)
- Empirical transition probability:
  - `P_hat(s'|s,a) = N(s,a,s') / N(s,a)`
- Empirical mean reward per transition:
  - `R_hat(s,a,s') = SumRewards(s,a,s') / N(s,a,s')`
- Bellman backup on empirical MDP:
  - `Q_k(s,a) = sum_{s'} P_hat(s'|s,a) * [R_hat(s,a,s') + gamma * V_{k-1}(s')]`
  - `V_k(s) = max_a Q_k(s,a)`
  - `pi(s) = argmax_a Q_k(s,a)` (tie-break by action id in code)
- Convergence criterion:
  - `residual_k = max_s |V_k(s) - V_{k-1}(s)|`
  - stop if `residual_k <= tolerance`.

### SARSA (`src/pacman_rldp/agents/sarsa.py`)
- On-policy TD update:
  - `Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]`

### Policy Iteration (`src/pacman_rldp/algorithms/policy_iteration/policy_iteration_obs.py`)
- Expected return for action from empirical outcomes:
  - `E(s,a) = sum_o p_o * [r_o + gamma * c_o * V(s_o')]`
  - with continuation factor `c_o = 1 - max(terminated_fraction_o, truncated_fraction_o)`
- Policy evaluation (iterative):
  - `V(s) <- E(s, pi(s))` until `max_s |V_new(s)-V_old(s)| < theta`.
- Policy improvement:
  - `pi_new(s) = argmax_a E(s,a)`.

## Tabular vs Non-Tabular Methods
| Method | Implementation | Data Representation |
|---|---|---|
| Baseline heuristic | `src/pacman_rldp/agents/baseline.py` | Non-tabular (rule-based) |
| Q-learning | `scripts/q_learning_agent.py` | Non-tabular (linear function approximation, weights over features) |
| Value Iteration (food-bitmask empirical) | `src/pacman_rldp/algorithms/food_bitmask_value_iteration.py` | Tabular over aggregated observation states (`value/policy/q` dictionaries) |
| SARSA | `src/pacman_rldp/agents/sarsa.py` | Tabular (`q_table` dictionary) |
| Policy Iteration (obs MDP) | `src/pacman_rldp/algorithms/policy_iteration/policy_iteration_obs.py` | Tabular (explicit empirical state-action-outcome model) |

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
