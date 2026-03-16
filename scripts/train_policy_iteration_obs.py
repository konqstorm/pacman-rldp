"""Train policy iteration on an empirical observation MDP."""

from __future__ import annotations

import argparse

from pacman_rldp.algorithms.policy_iteration.pi_runner import train_pi


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for observation-MDP policy iteration."""
    parser = argparse.ArgumentParser(description="Train policy iteration on observation MDP.")
    parser.add_argument("--config", default="configs/policy_iteration_obs.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override training output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of data collection episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override base random seed.")
    return parser.parse_args()


def main() -> None:
    """Execute empirical MDP collection + policy iteration."""
    args = parse_args()
    metrics = train_pi(
        config_path=args.config,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        log_every=100,
    )
    print(
        "Training complete. "
        f"Episodes={metrics['episodes']}, states={metrics['state_count']}, transitions={metrics['transition_count']}"
    )


if __name__ == "__main__":
    main()
