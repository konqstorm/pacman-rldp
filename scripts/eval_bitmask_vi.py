"""CLI wrapper for evaluating food-bitmask value iteration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from pacman_rldp.pipelines_food_bitmask_vi import eval_food_bitmask_value_iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a food-bitmask approximate VI model.")
    parser.add_argument("--config", default="configs/bitmask_value_iteration.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="Path to model artifact pickle.")
    parser.add_argument("--output-dir", default=None, help="Override evaluation output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of eval episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override base evaluation seed.")
    parser.add_argument("--render-mode", choices=["human", "ansi", "none"], default=None)
    parser.add_argument("--gif-title", default=None, help="Output GIF name (without or with .gif).")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = eval_food_bitmask_value_iteration(
        config_path=args.config,
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
        "Food-bitmask VI evaluation complete. "
        f"Episodes={metrics['episodes']}, win_rate={metrics['win_rate']:.3f}, "
        f"mean_return={metrics['mean_return']:.3f}, mean_steps={metrics['mean_steps']:.3f}"
    )
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Important metrics JSON: {result['important_metrics_path']}")
    if result['gif_path']:
        print(f"GIF saved to: {result['gif_path']}")


if __name__ == "__main__":
    main()
