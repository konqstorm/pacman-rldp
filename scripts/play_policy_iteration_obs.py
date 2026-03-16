"""Play Pacman using observation-MDP policy iteration artifacts."""

from __future__ import annotations

import argparse

from pacman_rldp.algorithms.policy_iteration.pi_runner import run_pi


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for play script."""
    parser = argparse.ArgumentParser(description="Play Pacman with observation-MDP policy.")
    parser.add_argument("--config", default="configs/policy_iteration_obs.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="Path to policy pickle.")
    parser.add_argument("--render-mode", choices=["human", "ansi"], default="human")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def main() -> None:
    """Execute play script with observation-MDP policy."""
    args = parse_args()
    run_pi(
        config_path=args.config,
        model_path=args.model,
        render_mode=args.render_mode,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
