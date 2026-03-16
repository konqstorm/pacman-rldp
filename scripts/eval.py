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
from pacman_rldp.agents import BaselineNearestFoodAvoidGhostPolicy
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
    parser.add_argument(
        "--policy",
        choices=["random", "baseline"],
        default=None,
        help="Evaluation policy to run. Defaults to config or 'random'.",
    )
    return parser.parse_args()


def resolve_eval_schedule(
    *,
    policy_name: str,
    args_episodes: int | None,
    args_seed: int | None,
    eval_cfg: dict[str, Any],
    env_seed: int,
) -> tuple[int, int]:
    """Resolve episode count and base seed for selected evaluation policy."""
    if policy_name == "baseline":
        episodes = int(args_episodes if args_episodes is not None else 200)
        base_seed = int(args_seed if args_seed is not None else 42)
        return episodes, base_seed

    episodes = int(args_episodes if args_episodes is not None else eval_cfg.get("episodes", 10))
    base_seed = int(args_seed if args_seed is not None else env_seed + 10_000)
    return episodes, base_seed


def build_episode_seeds(*, base_seed: int, episodes: int) -> list[int]:
    """Build deterministic per-episode seed schedule."""
    return [int(base_seed + episode_idx) for episode_idx in range(episodes)]


def run_episode(env: PacmanEnv, policy: Any, seed: int) -> tuple[float, bool, int]:
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
    policy_name = str(args.policy or eval_cfg.get("policy", "random"))

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

    episodes, base_seed = resolve_eval_schedule(
        policy_name=policy_name,
        args_episodes=args.episodes,
        args_seed=args.seed,
        eval_cfg=eval_cfg,
        env_seed=env_cfg.seed,
    )
    policy_seed = int(eval_cfg.get("policy_seed", env_cfg.seed))
    if policy_name == "baseline":
        policy = BaselineNearestFoodAvoidGhostPolicy()
    elif policy_name == "random":
        policy = RandomPolicy(seed=policy_seed)
    else:
        raise ValueError(f"Unsupported policy '{policy_name}'.")

    default_output = paths_cfg.get("eval_output_dir", "results/eval")
    output_dir = ensure_directory(args.output_dir or default_output)

    returns: list[float] = []
    wins: list[bool] = []
    steps: list[int] = []

    episode_seeds = build_episode_seeds(base_seed=base_seed, episodes=episodes)
    for episode_seed in episode_seeds:
        total_reward, did_win, episode_steps = run_episode(env, policy, seed=episode_seed)
        returns.append(total_reward)
        wins.append(did_win)
        steps.append(episode_steps)

    env.close()

    mean_return = float(sum(returns) / max(1, len(returns)))
    avg_episode_length = float(sum(steps) / max(1, len(steps)))
    metrics = {
        "episodes": episodes,
        "mean_return": mean_return,
        "avg_reward": mean_return,
        "avg_episode_length": avg_episode_length,
        "win_rate": float(sum(1 for value in wins if value) / max(1, len(wins))),
        "returns": returns,
        "wins": wins,
        "steps": steps,
        "policy": policy_name,
        "base_seed": base_seed,
        "episode_seeds": episode_seeds,
        "model_path": str(model_file),
        "model_summary": model_data.get("summary", {}),
    }
    save_json(metrics, output_dir / "eval_metrics.json")

    print(
        "Evaluation complete. "
        f"Episodes={episodes}, mean_return={metrics['mean_return']:.3f}, "
        f"win_rate={metrics['win_rate']:.3f}, avg_length={metrics['avg_episode_length']:.3f}"
    )
    print(f"Metrics saved to: {output_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
