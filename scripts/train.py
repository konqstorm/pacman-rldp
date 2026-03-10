"""Training scaffold for Pacman DP/RL experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt


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
from pacman_rldp.utils import ensure_directory, save_json, save_pickle, load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training scaffold."""
    parser = argparse.ArgumentParser(description="Train scaffold for Pacman DP/RL project.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override training output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def run_episode(env: PacmanEnv, policy: RandomPolicy, seed: int) -> tuple[float, bool, int]:
    """Run one full episode and return total reward, win flag, and step count."""
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


def save_learning_curve(returns: list[float], output_path: Path) -> None:
    """Persist reward-per-episode plot as PNG."""
    figure = plt.figure(figsize=(8, 4))
    axes = figure.add_subplot(111)
    axes.plot(range(1, len(returns) + 1), returns)
    axes.set_title("Episode Return")
    axes.set_xlabel("Episode")
    axes.set_ylabel("Return")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    """Execute the training scaffold pipeline."""
    args = parse_args()
    configure_logging()

    cfg = load_yaml(args.config)
    env_cfg_dict = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    paths_cfg = cfg.get("paths", {})

    if args.seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": args.seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    episodes = int(args.episodes if args.episodes is not None else train_cfg.get("episodes", 20))
    policy_seed = int(train_cfg.get("policy_seed", env_cfg.seed))
    policy = RandomPolicy(seed=policy_seed)

    default_output = paths_cfg.get("train_output_dir", "results/train")
    output_dir = ensure_directory(args.output_dir or default_output)

    returns: list[float] = []
    wins: list[bool] = []
    steps: list[int] = []

    for episode_idx in range(episodes):
        episode_seed = int(env_cfg.seed + episode_idx)
        total_reward, did_win, episode_steps = run_episode(env, policy, seed=episode_seed)
        returns.append(total_reward)
        wins.append(did_win)
        steps.append(episode_steps)

    env.close()

    mean_return = float(sum(returns) / max(1, len(returns)))
    win_rate = float(sum(1 for value in wins if value) / max(1, len(wins)))

    curve_path = output_dir / "learning_curve.png"
    save_learning_curve(returns, curve_path)

    with (output_dir / "episode_returns.csv").open("w", encoding="utf-8") as handle:
        handle.write("episode,return,win,steps\n")
        for idx, (ret, win, step_count) in enumerate(zip(returns, wins, steps), start=1):
            handle.write(f"{idx},{ret:.6f},{int(win)},{step_count}\n")

    metrics: dict[str, Any] = {
        "episodes": episodes,
        "mean_return": mean_return,
        "win_rate": win_rate,
        "returns": returns,
        "wins": wins,
        "steps": steps,
        "artifact_learning_curve": str(curve_path),
    }
    save_json(metrics, output_dir / "train_metrics.json")

    placeholder_model = {
        "model_type": "dp_placeholder",
        "policy": "random",
        "config": cfg,
        "summary": {
            "episodes": episodes,
            "mean_return": mean_return,
            "win_rate": win_rate,
        },
    }
    save_pickle(placeholder_model, output_dir / "model.pkl")

    print(f"Training complete. Episodes={episodes}, mean_return={mean_return:.3f}, win_rate={win_rate:.3f}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
