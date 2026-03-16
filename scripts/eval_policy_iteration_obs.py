"""Evaluate a policy trained on empirical observation MDP."""

from __future__ import annotations

import argparse

from pacman_rldp.algorithms.policy_iteration.pi_runner import eval_pi


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate observation-MDP policy iteration.")
    parser.add_argument("--config", default="configs/policy_iteration_obs.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="Path to policy pickle.")
    parser.add_argument("--output-dir", default=None, help="Override evaluation output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of eval episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override evaluation base seed.")
    return parser.parse_args()


def main() -> None:
    """Execute evaluation and persist metrics."""
    args = parse_args()
    metrics = eval_pi(
        config_path=args.config,
        model_path=args.model,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
    )
    print(
        "Evaluation complete. "
        f"Episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}, "
        f"win_rate={metrics['win_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
