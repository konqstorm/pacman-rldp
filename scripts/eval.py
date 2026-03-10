"""Evaluation scaffold for Pacman DP/RL experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


def _bootstrap_src_path() -> None:
    """Ensure local src directory is importable without editable install."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from pacman_rldp.agents import RandomPolicy
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.logging import configure_logging
from pacman_rldp.utils import ensure_directory, load_pickle, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation scaffold."""
    parser = argparse.ArgumentParser(description="Evaluate a saved Pacman model artifact.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="Path to model artifact pickle.")
    parser.add_argument("--output-dir", default=None, help="Override evaluation output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of eval episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override evaluation seed.")
    return parser.parse_args()


def run_episode(env: PacmanEnv, policy: RandomPolicy, seed: int) -> tuple[float, bool, int]:
    """Run one evaluation episode and return summary metrics."""
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0

    while True:
        action = policy.select_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        if terminated or truncated:
            return total_reward, bool(info.get("is_win", False)), step_count


def main() -> None:
    """Execute evaluation scaffold and persist metrics."""
    args = parse_args()
    configure_logging()

    cfg = load_yaml(args.config)
    env_cfg_dict = cfg.get("env", {})
    eval_cfg = cfg.get("eval", {})
    paths_cfg = cfg.get("paths", {})

    if args.seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": args.seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    model_path = args.model or paths_cfg.get("model_path", "results/train/model.pkl")
    model_data: dict[str, Any] = {}
    model_file = Path(model_path)
    if model_file.exists():
        loaded = load_pickle(model_file)
        if isinstance(loaded, dict):
            model_data = loaded

    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 10))
    policy_seed = int(eval_cfg.get("policy_seed", env_cfg.seed))
    policy = RandomPolicy(seed=policy_seed)

    default_output = paths_cfg.get("eval_output_dir", "results/eval")
    output_dir = ensure_directory(args.output_dir or default_output)

    returns: list[float] = []
    wins: list[bool] = []
    steps: list[int] = []

    for episode_idx in range(episodes):
        episode_seed = int(env_cfg.seed + 10_000 + episode_idx)
        total_reward, did_win, episode_steps = run_episode(env, policy, seed=episode_seed)
        returns.append(total_reward)
        wins.append(did_win)
        steps.append(episode_steps)

    env.close()

    metrics = {
        "episodes": episodes,
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for value in wins if value) / max(1, len(wins))),
        "returns": returns,
        "wins": wins,
        "steps": steps,
        "model_path": str(model_file),
        "model_summary": model_data.get("summary", {}),
    }
    save_json(metrics, output_dir / "eval_metrics.json")

    print(
        "Evaluation complete. "
        f"Episodes={episodes}, mean_return={metrics['mean_return']:.3f}, win_rate={metrics['win_rate']:.3f}"
    )
    print(f"Metrics saved to: {output_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
