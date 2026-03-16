"""Unified training entrypoint for multiple algorithms."""

from __future__ import annotations

import argparse

from pacman_rldp.algorithms.policy_iteration.pi_runner import train_pi
from pacman_rldp.pipelines_food_bitmask_vi import train_food_bitmask_value_iteration
from pacman_rldp.pipelines_tabular_q import train_q_learning, train_sarsa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified training script.")
    parser.add_argument("algo", choices=["pi", "vi", "q_learning", "sarsa"], help="Algorithm name")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate (q/sarsa)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon-greedy (q/sarsa)")
    parser.add_argument("--collection-episodes", type=int, default=None, help="VI collection episodes")
    parser.add_argument("--max-iterations", type=int, default=None, help="VI max iterations")
    parser.add_argument("--tolerance", type=float, default=None, help="VI tolerance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.algo == "pi":
        config = args.config or "configs/policy_iteration_obs.yaml"
        metrics = train_pi(
            config_path=config,
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
            log_every=100,
        )
        print(
            "PI training complete. "
            f"Episodes={metrics['episodes']}, states={metrics['state_count']}, transitions={metrics['transition_count']}"
        )
        return

    if args.algo == "vi":
        config = args.config or "configs/bitmask_value_iteration.yaml"
        result = train_food_bitmask_value_iteration(
            config_path=config,
            output_dir=args.output_dir,
            collection_episodes=args.collection_episodes,
            gamma=args.gamma,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            seed=args.seed,
        )
        metrics = result["metrics"]
        print(
            "VI training complete. "
            f"states={metrics['discovered_states']}, iterations={metrics['iterations']}, "
            f"total_seconds={metrics['total_seconds']:.3f}"
        )
        return

    if args.algo == "q_learning":
        config = args.config or "configs/default.yaml"
        metrics = train_q_learning(
            config_path=config,
            output_dir=args.output_dir,
            episodes=args.episodes or 1000,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.seed,
        )
        print(
            "Q-learning training complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}"
        )
        return

    if args.algo == "sarsa":
        config = args.config or "configs/default.yaml"
        metrics = train_sarsa(
            config_path=config,
            output_dir=args.output_dir,
            episodes=args.episodes or 1000,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            seed=args.seed,
        )
        print(
            "SARSA training complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}"
        )
        return


if __name__ == "__main__":
    main()
