"""Unified play entrypoint for multiple algorithms."""

from __future__ import annotations

import argparse

from pacman_rldp.algorithms.policy_iteration.pi_runner import run_pi
from pacman_rldp.pipelines_food_bitmask_vi import eval_food_bitmask_value_iteration
from pacman_rldp.pipelines_tabular_q import run_tabular_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified play script.")
    parser.add_argument("algo", choices=["pi", "vi", "q_learning", "sarsa"], help="Algorithm name")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--model", default=None, help="Path to model artifact")
    parser.add_argument("--render-mode", default="human", help="Render mode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--gif-title", default=None, help="GIF title (VI only)")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF (VI only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.algo == "pi":
        config = args.config or "configs/policy_iteration_obs.yaml"
        run_pi(
            config_path=config,
            model_path=args.model,
            render_mode=args.render_mode,
            episodes=args.episodes,
            seed=args.seed,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        return

    if args.algo == "vi":
        config = args.config or "configs/bitmask_value_iteration.yaml"
        eval_food_bitmask_value_iteration(
            config_path=config,
            model_path=args.model,
            output_dir=None,
            episodes=args.episodes,
            seed=args.seed,
            render_mode=args.render_mode,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        return

    if args.algo in {"q_learning", "sarsa"}:
        config = args.config or "configs/default.yaml"
        default_model = (
            "results/train_q_learning/q_table.pkl" if args.algo == "q_learning" else "results/train_sarsa/q_table.pkl"
        )
        run_tabular_q(
            config_path=config,
            model_path=args.model or default_model,
            render_mode=args.render_mode,
            episodes=args.episodes,
            seed=args.seed,
            gif_title=args.gif_title,
            no_gif=args.no_gif,
        )
        return


if __name__ == "__main__":
    main()
