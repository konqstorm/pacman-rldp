"""CLI wrapper for training food-bitmask value iteration."""

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

from pacman_rldp.pipelines_food_bitmask_vi import train_food_bitmask_value_iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train food-bitmask approximate value iteration model.")
    parser.add_argument("--config", default="configs/bitmask_value_iteration.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override training output directory.")
    parser.add_argument("--collection-episodes", type=int, default=None, help="Override number of data-collection episodes.")
    parser.add_argument("--gamma", type=float, default=None, help="Override VI discount factor.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max VI iterations.")
    parser.add_argument("--tolerance", type=float, default=None, help="Override VI convergence tolerance.")
    parser.add_argument("--seed", type=int, default=None, help="Override base collection seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_food_bitmask_value_iteration(
        config_path=args.config,
        output_dir=args.output_dir,
        collection_episodes=args.collection_episodes,
        gamma=args.gamma,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    metrics = result["metrics"]
    print(
        "Food-bitmask VI training complete. "
        f"states={metrics['discovered_states']}, iterations={metrics['iterations']}, "
        f"total_seconds={metrics['total_seconds']:.3f}"
    )
    print(f"Artifacts saved to: {args.output_dir or 'results/train_food_bitmask_vi'}")
    print(f"Important training reward plot: {result['important_reward_plot_path']}")
    print(f"Important metrics JSON: {result['important_metrics_path']}")


if __name__ == "__main__":
    main()
