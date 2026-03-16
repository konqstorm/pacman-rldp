"""Train policy iteration on an empirical observation MDP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pacman_rldp.algorithms.policy_iteration import (
    ObsMDPModel,
    PolicyIterationResult,
    encode_observation,
    policy_iteration,
)
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.logging import configure_logging
from pacman_rldp.utils import ensure_directory, load_yaml, save_json, save_pickle

_OBS_DIR_TO_ACTION = {1: 0, 2: 1, 3: 2, 4: 3}
_OBS_DIR_OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for observation-MDP policy iteration."""
    parser = argparse.ArgumentParser(description="Train policy iteration on observation MDP.")
    parser.add_argument("--config", default="configs/policy_iteration_obs.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override training output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of data collection episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override base random seed.")
    return parser.parse_args()


def save_learning_curve(returns: list[float], output_path: Path) -> None:
    """Persist reward-per-episode plot as PNG."""
    figure = plt.figure(figsize=(8, 4))
    axes = figure.add_subplot(111)
    axes.plot(range(1, len(returns) + 1), returns)
    axes.set_title("Episode Return (Data Collection)")
    axes.set_xlabel("Episode")
    axes.set_ylabel("Return")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def run_episode(
    env: PacmanEnv,
    rng: np.random.Generator,
    model: ObsMDPModel,
    *,
    seed: int,
    drop_keys: list[str],
    float_round: int,
) -> tuple[float, bool, int]:
    """Collect one episode of transitions into the empirical model."""
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0

    while True:
        action = choose_action_heuristic(observation, info, rng, env)

        next_observation, reward, terminated, truncated, next_info = env.step(action)

        model.update(
            encode_observation(observation, drop_keys=drop_keys, float_round=float_round),
            action,
            float(reward),
            encode_observation(next_observation, drop_keys=drop_keys, float_round=float_round),
            bool(terminated),
            bool(truncated),
        )

        total_reward += float(reward)
        step_count += 1
        observation = next_observation
        info = next_info

        if terminated or truncated:
            return total_reward, bool(info.get("is_win", False)), step_count


def choose_action_heuristic(
    observation: dict[str, Any],
    info: dict[str, Any],
    rng: np.random.Generator,
    env: PacmanEnv,
) -> int:
    """Simple heuristic: avoid nearby ghost, otherwise move toward nearest food."""
    legal_actions = info.get("legal_action_ids", [])
    if not legal_actions:
        return int(env.action_space.sample())

    ghost_bucket = int(observation.get("nearest_ghost_bucket", np.array([-1]))[0])
    ghost_dir = int(observation.get("nearest_ghost_direction", np.array([0]))[0])
    food_dir = int(observation.get("nearest_food_direction", np.array([0]))[0])

    if ghost_bucket >= 0 and ghost_bucket <= 1 and ghost_dir in _OBS_DIR_OPPOSITE:
        avoid_dir = _OBS_DIR_OPPOSITE[ghost_dir]
        avoid_action = _OBS_DIR_TO_ACTION.get(avoid_dir)
        if avoid_action in legal_actions:
            return int(avoid_action)

    food_action = _OBS_DIR_TO_ACTION.get(food_dir)
    if food_action in legal_actions:
        return int(food_action)

    return int(rng.choice(legal_actions))


def main() -> None:
    """Execute empirical MDP collection + policy iteration."""
    args = parse_args()
    configure_logging()

    cfg = load_yaml(args.config)
    env_cfg_dict = cfg.get("env", {})
    obs_mdp_cfg = cfg.get("obs_mdp", {})
    pi_cfg = cfg.get("policy_iteration", {})
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if args.seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": args.seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    episodes = int(args.episodes if args.episodes is not None else obs_mdp_cfg.get("episodes", 60000))
    base_seed = int(obs_mdp_cfg.get("seed_base", env_cfg.seed))

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    rng_seed = int(obs_mdp_cfg.get("policy_seed", env_cfg.seed))
    rng = np.random.default_rng(rng_seed)

    model = ObsMDPModel()

    default_output = paths_cfg.get("train_output_dir", "results/obs_policy_iteration")
    output_dir = ensure_directory(args.output_dir or default_output)

    returns: list[float] = []
    wins: list[bool] = []
    steps: list[int] = []

    print(f"Collecting empirical MDP for {episodes} episodes...")
    for episode_idx in range(episodes):
        episode_seed = int(base_seed + episode_idx)
        total_reward, did_win, episode_steps = run_episode(
            env,
            rng,
            model,
            seed=episode_seed,
            drop_keys=drop_keys,
            float_round=float_round,
        )
        returns.append(total_reward)
        wins.append(did_win)
        steps.append(episode_steps)
        if (episode_idx + 1) % 100 == 0:
            win_rate = sum(1 for value in wins if value) / max(1, len(wins))
            print(
                "Collected "
                f"{episode_idx + 1}/{episodes} episodes | wins={sum(wins)} | win_rate={win_rate:.4f}"
            )

    env.close()

    model_path = output_dir / "empirical_mdp.pkl"
    save_pickle(model, model_path)

    print("Running policy iteration on empirical MDP...")
    pi_result: PolicyIterationResult = policy_iteration(
        model,
        gamma=float(pi_cfg.get("gamma", 0.95)),
        theta=float(pi_cfg.get("theta", 1e-4)),
        max_eval_iters=int(pi_cfg.get("max_eval_iters", 50)),
        max_policy_iters=int(pi_cfg.get("max_policy_iters", 50)),
    )
    print(
        "Policy iteration complete. "
        f"iterations={pi_result.policy_iterations}, eval_sweeps={pi_result.evaluation_sweeps}"
    )

    policy_path = output_dir / "policy.pkl"
    values_path = output_dir / "values.pkl"
    save_pickle(pi_result.policy, policy_path)
    save_pickle(pi_result.values, values_path)

    curve_path = output_dir / "data_collection_returns.png"
    save_learning_curve(returns, curve_path)

    mean_return = float(sum(returns) / max(1, len(returns)))
    win_rate = float(sum(1 for value in wins if value) / max(1, len(wins)))

    metrics: dict[str, Any] = {
        "episodes": episodes,
        "mean_return": mean_return,
        "win_rate": win_rate,
        "avg_steps": float(sum(steps) / max(1, len(steps))),
        "state_count": len(model.states()),
        "state_action_count": model.state_action_count(),
        "transition_count": model.transition_count(),
        "policy_iterations": pi_result.policy_iterations,
        "evaluation_sweeps": pi_result.evaluation_sweeps,
        "last_delta": pi_result.last_delta,
        "artifact_empirical_mdp": str(model_path),
        "artifact_policy": str(policy_path),
        "artifact_values": str(values_path),
        "artifact_returns_plot": str(curve_path),
    }
    save_json(metrics, output_dir / "train_metrics.json")

    print(
        "Training complete. "
        f"Episodes={episodes}, states={metrics['state_count']}, transitions={metrics['transition_count']}"
    )
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
