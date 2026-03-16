"""Simple tabular Q-learning and SARSA pipelines over encoded observations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageGrab

from .algorithms.policy_iteration.obs_encoding import encode_observation
from .env import PacmanEnv, build_env_config
from .logging import configure_logging
from .third_party.bk import graphicsUtils
from .utils import ensure_directory, load_pickle, load_yaml, save_json, save_pickle


class TabularAgent:
    """Base tabular agent with epsilon-greedy behavior."""

    def __init__(self, action_size: int, alpha: float, gamma: float, epsilon: float) -> None:
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict[Any, np.ndarray] = {}
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _get_q(self, state_key: Any) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state_key]

    def select_action(self, state_key: Any, legal_actions: list[int]) -> int:
        if not legal_actions:
            return int(self._rng.integers(0, self.action_size))
        if self._rng.random() < self.epsilon:
            return int(self._rng.choice(legal_actions))
        q_values = self._get_q(state_key)
        masked = np.full(self.action_size, -np.inf, dtype=np.float32)
        for act in legal_actions:
            masked[act] = q_values[act]
        best = np.flatnonzero(masked == np.max(masked))
        return int(self._rng.choice(best))


class QLearningAgent(TabularAgent):
    def update(self, state_key: Any, action: int, reward: float, next_state_key: Any, next_legal: list[int], done: bool) -> None:
        q_values = self._get_q(state_key)
        current = float(q_values[action])
        if done or not next_legal:
            target = reward
        else:
            next_q = self._get_q(next_state_key)
            target = reward + self.gamma * float(np.max(next_q[next_legal]))
        q_values[action] = current + self.alpha * (target - current)


class SarsaAgent(TabularAgent):
    def update(
        self,
        state_key: Any,
        action: int,
        reward: float,
        next_state_key: Any,
        next_action: int | None,
        done: bool,
    ) -> None:
        q_values = self._get_q(state_key)
        current = float(q_values[action])
        if done or next_action is None:
            target = reward
        else:
            next_q = self._get_q(next_state_key)
            target = reward + self.gamma * float(next_q[next_action])
        q_values[action] = current + self.alpha * (target - current)


def _encode_obs(observation: dict[str, Any], drop_keys: list[str], float_round: int) -> Any:
    return encode_observation(observation, drop_keys=drop_keys, float_round=float_round)


def train_q_learning(
    *,
    config_path: str = "configs/default.yaml",
    output_dir: str | None = None,
    episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    agent = QLearningAgent(env.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon)
    agent.seed(env_cfg.seed)

    returns: list[float] = []
    wins: list[bool] = []

    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        total_reward = 0.0
        while True:
            state_key = _encode_obs(observation, drop_keys, float_round)
            legal_actions = info.get("legal_action_ids", [])
            action = agent.select_action(state_key, legal_actions)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)
            next_state_key = _encode_obs(next_obs, drop_keys, float_round)
            next_legal = next_info.get("legal_action_ids", [])
            agent.update(state_key, action, float(reward), next_state_key, next_legal, done)
            total_reward += float(reward)
            observation = next_obs
            info = next_info
            if done:
                wins.append(bool(info.get("is_win", False)))
                returns.append(total_reward)
                break

    env.close()

    default_output = paths_cfg.get("train_output_dir", "results/train_q_learning")
    output_dir_path = ensure_directory(output_dir or default_output)
    model_path = output_dir_path / "q_table.pkl"
    save_pickle(agent.q_table, model_path)

    metrics = {
        "episodes": episodes,
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for v in wins if v) / max(1, len(wins))),
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "model_path": str(model_path),
    }
    save_json(metrics, output_dir_path / "train_q_learning_metrics.json")
    return metrics


def train_sarsa(
    *,
    config_path: str = "configs/default.yaml",
    output_dir: str | None = None,
    episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    paths_cfg = cfg.get("paths", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    agent = SarsaAgent(env.action_space.n, alpha=alpha, gamma=gamma, epsilon=epsilon)
    agent.seed(env_cfg.seed)

    returns: list[float] = []
    wins: list[bool] = []

    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        total_reward = 0.0
        state_key = _encode_obs(observation, drop_keys, float_round)
        legal_actions = info.get("legal_action_ids", [])
        action = agent.select_action(state_key, legal_actions)

        while True:
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)
            next_state_key = _encode_obs(next_obs, drop_keys, float_round)
            next_legal = next_info.get("legal_action_ids", [])
            next_action = agent.select_action(next_state_key, next_legal) if not done else None
            agent.update(state_key, action, float(reward), next_state_key, next_action, done)

            total_reward += float(reward)
            observation = next_obs
            info = next_info
            state_key = next_state_key
            action = next_action if next_action is not None else 0

            if done:
                wins.append(bool(info.get("is_win", False)))
                returns.append(total_reward)
                break

    env.close()

    default_output = paths_cfg.get("train_output_dir", "results/train_sarsa")
    output_dir_path = ensure_directory(output_dir or default_output)
    model_path = output_dir_path / "q_table.pkl"
    save_pickle(agent.q_table, model_path)

    metrics = {
        "episodes": episodes,
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for v in wins if v) / max(1, len(wins))),
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "model_path": str(model_path),
    }
    save_json(metrics, output_dir_path / "train_sarsa_metrics.json")
    return metrics


def eval_tabular_q(
    *,
    config_path: str = "configs/default.yaml",
    model_path: str,
    output_dir: str | None = None,
    episodes: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    configure_logging()
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}

    env_cfg = build_env_config(env_cfg_dict)
    env = PacmanEnv(config=env_cfg, render_mode=None)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    q_table = load_pickle(model_path)

    returns: list[float] = []
    wins: list[bool] = []

    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        total_reward = 0.0
        while True:
            state_key = _encode_obs(observation, drop_keys, float_round)
            q_values = q_table.get(state_key)
            legal_actions = info.get("legal_action_ids", [])
            if q_values is None or not legal_actions:
                action = int(np.random.default_rng().choice(legal_actions)) if legal_actions else int(env.action_space.sample())
            else:
                masked = np.full(env.action_space.n, -np.inf, dtype=np.float32)
                for act in legal_actions:
                    masked[act] = q_values[act]
                best = np.flatnonzero(masked == np.max(masked))
                action = int(np.random.default_rng().choice(best))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                returns.append(total_reward)
                wins.append(bool(info.get("is_win", False)))
                break

    env.close()

    default_output = "results/eval_tabular_q"
    output_dir_path = ensure_directory(output_dir or default_output)

    metrics = {
        "episodes": episodes,
        "mean_return": float(sum(returns) / max(1, len(returns))),
        "win_rate": float(sum(1 for v in wins if v) / max(1, len(wins))),
        "model_path": str(model_path),
    }
    save_json(metrics, output_dir_path / "eval_tabular_q_metrics.json")
    return metrics


def run_tabular_q(
    *,
    config_path: str = "configs/default.yaml",
    model_path: str,
    render_mode: str = "human",
    episodes: int = 1,
    seed: int | None = None,
    gif_title: str | None = None,
    no_gif: bool = False,
) -> None:
    cfg = load_yaml(config_path)
    env_cfg_dict = cfg.get("env", {})
    encoding_cfg = cfg.get("obs_encoding", {})

    if seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": seed}

    env_cfg = build_env_config(env_cfg_dict)
    save_gif_enabled = (not no_gif) and (render_mode == "human")
    gif_path: Path | None = None
    if save_gif_enabled:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tabular_q_{stamp}.gif" if not gif_title else (gif_title if gif_title.endswith(".gif") else f"{gif_title}.gif")
        gif_dir = ensure_directory("results/important")
        gif_path = gif_dir / filename

    env = PacmanEnv(config=env_cfg, render_mode=render_mode)

    drop_keys = list(encoding_cfg.get("drop_keys", ["score", "step_count"]))
    float_round = int(encoding_cfg.get("float_round", 3))

    q_table = load_pickle(model_path)

    gif_frames: list[Image.Image] = []
    for episode_idx in range(episodes):
        observation, info = env.reset(seed=env_cfg.seed + episode_idx)
        if save_gif_enabled:
            frame = _capture_human_frame()
            if frame is not None:
                gif_frames.append(frame)
        while True:
            state_key = _encode_obs(observation, drop_keys, float_round)
            q_values = q_table.get(state_key)
            legal_actions = info.get("legal_action_ids", [])
            if q_values is None or not legal_actions:
                action = int(np.random.default_rng().choice(legal_actions)) if legal_actions else int(env.action_space.sample())
            else:
                masked = np.full(env.action_space.n, -np.inf, dtype=np.float32)
                for act in legal_actions:
                    masked[act] = q_values[act]
                best = np.flatnonzero(masked == np.max(masked))
                action = int(np.random.default_rng().choice(best))
            observation, reward, terminated, truncated, info = env.step(action)
            if save_gif_enabled:
                frame = _capture_human_frame()
                if frame is not None:
                    gif_frames.append(frame)
            if terminated or truncated:
                break

    env.close()
    if save_gif_enabled and gif_path is not None:
        _save_gif(gif_frames, gif_path, frame_time=env_cfg.frame_time)
        print(f"Saved GIF: {gif_path}")


def _capture_human_frame() -> Image.Image | None:
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


def _save_gif(frames: list[Image.Image], output_path: Path, frame_time: float) -> None:
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
