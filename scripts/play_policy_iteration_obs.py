"""Play Pacman using observation-MDP policy iteration artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from PIL import Image, ImageGrab

from pacman_rldp.agents import ObsPolicy
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.third_party.bk import graphicsUtils
from pacman_rldp.utils import ensure_directory, load_pickle, load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for play script."""
    parser = argparse.ArgumentParser(description="Play Pacman with observation-MDP policy.")
    parser.add_argument("--config", default="configs/policy_iteration_obs.yaml", help="Path to YAML config.")
    parser.add_argument("--model", default=None, help="Path to policy pickle.")
    parser.add_argument("--render-mode", choices=["human", "ansi"], default="human")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--gif-title", default=None, help="Output GIF name (without or with .gif).")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF export.")
    return parser.parse_args()


def build_gif_filename(gif_title: str | None) -> str:
    """Resolve GIF filename from user-provided title or default timestamped name."""
    if gif_title is None or not gif_title.strip():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"experiment_{stamp}.gif"
    normalized = gif_title.strip()
    if not normalized.lower().endswith(".gif"):
        normalized = f"{normalized}.gif"
    return normalized


def capture_human_frame() -> Image.Image | None:
    """Capture one frame from the active Tk canvas used by human visualization."""
    canvas = graphicsUtils._canvas
    root_window = graphicsUtils._root_window
    if canvas is None or root_window is None:
        return None

    root_window.update_idletasks()
    root_window.update()
    x0 = int(canvas.winfo_rootx())
    y0 = int(canvas.winfo_rooty())
    width = int(canvas.winfo_width())
    height = int(canvas.winfo_height())
    if width <= 1 or height <= 1:
        return None
    return ImageGrab.grab(bbox=(x0, y0, x0 + width, y0 + height))


def save_gif(frames: list[Image.Image], output_path: Path, frame_time: float) -> None:
    """Persist accumulated frames into an animated GIF file."""
    if not frames:
        return
    duration_ms = max(20, int(frame_time * 1000))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )


def run_env_loop(
    env: PacmanEnv,
    policy: ObsPolicy,
    episodes: int,
    seed: int,
    save_human_gif_frames: bool,
) -> list[Image.Image]:
    """Run policy in environment for a fixed number of episodes."""
    gif_frames: list[Image.Image] = []
    for episode_idx in range(episodes):
        observation, info = env.reset(seed=seed + episode_idx)
        if save_human_gif_frames:
            frame = capture_human_frame()
            if frame is not None:
                gif_frames.append(frame)
        print(f"Episode {episode_idx + 1} started.")
        while True:
            action = policy.select_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            if save_human_gif_frames:
                frame = capture_human_frame()
                if frame is not None:
                    gif_frames.append(frame)
            if env.render_mode == "ansi":
                rendered = env.render()
                if rendered is not None:
                    print(rendered)
                print(f"Reward: {reward:.2f} | Score: {info.get('score', 0.0):.2f}")
            if terminated or truncated:
                print(
                    f"Episode {episode_idx + 1} finished. "
                    f"win={info.get('is_win', False)} score={info.get('score', 0.0):.2f}"
                )
                break
    return gif_frames


def main() -> None:
    """Execute play script with observation-MDP policy."""
    args = parse_args()
    cfg = load_yaml(args.config)
    env_cfg_dict = cfg.get("env", {})
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if args.seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": args.seed}

    env_cfg = build_env_config(env_cfg_dict)

    model_path = args.model or paths_cfg.get("model_path", "results/obs_policy_iteration/policy.pkl")
    policy_table = load_pickle(Path(model_path))

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))
    policy = ObsPolicy(policy_table, seed=env_cfg.seed, drop_keys=drop_keys, float_round=float_round)

    save_gif_enabled = (not args.no_gif) and (args.render_mode == "human")
    gif_path: Path | None = None
    if save_gif_enabled:
        gif_filename = build_gif_filename(args.gif_title)
        gif_dir = ensure_directory("results/important")
        gif_path = gif_dir / gif_filename
    elif (not args.no_gif) and args.render_mode != "human":
        print("GIF export is available only for render-mode=human. Skipping GIF save.")

    env = PacmanEnv(config=env_cfg, render_mode=args.render_mode)
    gif_frames = run_env_loop(
        env=env,
        policy=policy,
        episodes=args.episodes,
        seed=env_cfg.seed,
        save_human_gif_frames=save_gif_enabled,
    )
    env.close()
    if save_gif_enabled and gif_path is not None:
        save_gif(gif_frames, gif_path, frame_time=env_cfg.frame_time)
        print(f"Saved GIF: {gif_path}")


if __name__ == "__main__":
    main()
