"""Unified evaluation entrypoint for multiple algorithms."""

from __future__ import annotations

import argparse

from pacman_rldp.algorithms.policy_iteration.pi_runner import eval_pi
from pacman_rldp.pipelines_food_bitmask_vi import eval_food_bitmask_value_iteration
from pacman_rldp.pipelines_tabular_q import eval_tabular_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation script.")
    parser.add_argument("algo", choices=["pi", "vi", "q_learning", "sarsa"], help="Algorithm name")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--model", default=None, help="Path to model artifact")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--render-mode", default=None, help="Render mode for VI eval")
    parser.add_argument("--gif-title", default=None, help="GIF title for VI eval")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF for VI eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.algo == "pi":
        config = args.config or "configs/policy_iteration_obs.yaml"
        metrics = eval_pi(
            config_path=config,
            model_path=args.model,
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
        )
        print(
            "PI evaluation complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}, "
            f"win_rate={metrics['win_rate']:.3f}"
        )
        return

    if args.algo == "vi":
        config = args.config or "configs/bitmask_value_iteration.yaml"
        result = eval_food_bitmask_value_iteration(
            config_path=config,
            model_path=args.model,
            output_dir=args.output_dir,
            episodes=args.episodes,
            seed=args.seed,
            render_mode=args.render_mode,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        metrics = result["metrics"]
        print(
            "VI evaluation complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}, "
            f"win_rate={metrics['win_rate']:.3f}"
        )
        return

    if args.algo in {"q_learning", "sarsa"}:
        config = args.config or "configs/default.yaml"
        default_model = (
            "results/important/q_weights.pkl" if args.algo == "q_learning" else "results/train_sarsa/q_table.pkl"
        )
        metrics = eval_tabular_q(
            config_path=config,
            model_path=args.model or default_model,
            output_dir=args.output_dir,
            episodes=args.episodes or 200,
            seed=args.seed,
        )
        print(
            f"{args.algo} evaluation complete. "
            f"episodes={metrics['episodes']}, mean_return={metrics['mean_return']:.3f}, "
            f"win_rate={metrics['win_rate']:.3f}"
        )
        return


if __name__ == "__main__":
    main()
