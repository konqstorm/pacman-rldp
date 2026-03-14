"""Training scaffold for Pacman DP/RL experiments."""

from __future__ import annotations

# import argparse
# from pathlib import Path
# import sys
# from typing import Any

# import matplotlib.pyplot as plt


# def _bootstrap_src_path() -> None:
#     """Ensure local src directory is importable without editable install."""
#     project_root = Path(__file__).resolve().parents[1]
#     src_path = project_root / "src"
#     if str(src_path) not in sys.path:
#         sys.path.insert(0, str(src_path))


# _bootstrap_src_path()

# from pacman_rldp.agents import RandomPolicy
# from pacman_rldp.env import PacmanEnv, build_env_config
# from pacman_rldp.logging import configure_logging
# from pacman_rldp.utils import ensure_directory, save_json, save_pickle, load_yaml


# def parse_args() -> argparse.Namespace:
#     """Parse command-line arguments for the training scaffold."""
#     parser = argparse.ArgumentParser(description="Train scaffold for Pacman DP/RL project.")
#     parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
#     parser.add_argument("--output-dir", default=None, help="Override training output directory.")
#     parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes.")
#     parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
#     return parser.parse_args()


# def run_episode(env: PacmanEnv, policy: RandomPolicy, seed: int) -> tuple[float, bool, int]:
#     """Run one full episode and return total reward, win flag, and step count."""
#     observation, info = env.reset(seed=seed)
#     total_reward = 0.0
#     step_count = 0

#     while True:
#         action = policy.select_action(observation, info)
#         observation, reward, terminated, truncated, info = env.step(action)
#         total_reward += float(reward)
#         step_count += 1
#         if terminated or truncated:
#             return total_reward, bool(info.get("is_win", False)), step_count


# def save_learning_curve(returns: list[float], output_path: Path) -> None:
#     """Persist reward-per-episode plot as PNG."""
#     figure = plt.figure(figsize=(8, 4))
#     axes = figure.add_subplot(111)
#     axes.plot(range(1, len(returns) + 1), returns)
#     axes.set_title("Episode Return")
#     axes.set_xlabel("Episode")
#     axes.set_ylabel("Return")
#     axes.grid(True, alpha=0.3)
#     figure.tight_layout()
#     figure.savefig(output_path)
#     plt.close(figure)


# def main() -> None:
#     """Execute the training scaffold pipeline."""
#     args = parse_args()
#     configure_logging()

#     cfg = load_yaml(args.config)
#     env_cfg_dict = cfg.get("env", {})
#     train_cfg = cfg.get("train", {})
#     paths_cfg = cfg.get("paths", {})

#     if args.seed is not None:
#         env_cfg_dict = {**env_cfg_dict, "seed": args.seed}

#     env_cfg = build_env_config(env_cfg_dict)
#     env = PacmanEnv(config=env_cfg, render_mode=None)

#     episodes = int(args.episodes if args.episodes is not None else train_cfg.get("episodes", 20))
#     policy_seed = int(train_cfg.get("policy_seed", env_cfg.seed))
#     policy = RandomPolicy(seed=policy_seed)

#     default_output = paths_cfg.get("train_output_dir", "results/train")
#     output_dir = ensure_directory(args.output_dir or default_output)

#     returns: list[float] = []
#     wins: list[bool] = []
#     steps: list[int] = []

#     for episode_idx in range(episodes):
#         episode_seed = int(env_cfg.seed + episode_idx)
#         total_reward, did_win, episode_steps = run_episode(env, policy, seed=episode_seed)
#         returns.append(total_reward)
#         wins.append(did_win)
#         steps.append(episode_steps)

#     env.close()

#     mean_return = float(sum(returns) / max(1, len(returns)))
#     win_rate = float(sum(1 for value in wins if value) / max(1, len(wins)))

#     curve_path = output_dir / "learning_curve.png"
#     save_learning_curve(returns, curve_path)

#     with (output_dir / "episode_returns.csv").open("w", encoding="utf-8") as handle:
#         handle.write("episode,return,win,steps\n")
#         for idx, (ret, win, step_count) in enumerate(zip(returns, wins, steps), start=1):
#             handle.write(f"{idx},{ret:.6f},{int(win)},{step_count}\n")

#     metrics: dict[str, Any] = {
#         "episodes": episodes,
#         "mean_return": mean_return,
#         "win_rate": win_rate,
#         "returns": returns,
#         "wins": wins,
#         "steps": steps,
#         "artifact_learning_curve": str(curve_path),
#     }
#     save_json(metrics, output_dir / "train_metrics.json")

#     placeholder_model = {
#         "model_type": "dp_placeholder",
#         "policy": "random",
#         "config": cfg,
#         "summary": {
#             "episodes": episodes,
#             "mean_return": mean_return,
#             "win_rate": win_rate,
#         },
#     }
#     save_pickle(placeholder_model, output_dir / "model.pkl")

#     print(f"Training complete. Episodes={episodes}, mean_return={mean_return:.3f}, win_rate={win_rate:.3f}")
#     print(f"Artifacts saved to: {output_dir}")


# if __name__ == "__main__":
#     main()


# import argparse
# import yaml

# import argparse
# from pathlib import Path
# import sys
# from typing import Any

# import matplotlib.pyplot as plt
# import numpy as np

# def _bootstrap_src_path() -> None:
#     """Ensure local src directory is importable without editable install."""
#     project_root = Path(__file__).resolve().parents[1]
#     src_path = project_root / "src"
#     if str(src_path) not in sys.path:
#         sys.path.insert(0, str(src_path))

# _bootstrap_src_path()

# from pacman_rldp.agents import RandomPolicy
# from pacman_rldp.agents.sarsa import SarsaAgent 
# from pacman_rldp.env import PacmanEnv, build_env_config
# from pacman_rldp.logging import configure_logging
# from pacman_rldp.utils import ensure_directory, save_json, save_pickle, load_yaml

# AGENT_REGISTRY = {
#     "random": RandomPolicy,
#     "sarsa": SarsaAgent,
#     # "q_learning": QLearningAgent,  # Легко добавить в будущем
# }

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Train scaffold for Pacman DP/RL project.")
#     parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
#     parser.add_argument("--output-dir", default=None, help="Override training output directory.")
#     parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes.")
#     parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
#     return parser.parse_args()


# def train_episode(env: PacmanEnv, agent: Any, seed: int) -> tuple[float, bool, int]:
#     """Run one training episode including the agent.update() step."""
#     observation, info = env.reset(seed=seed)
#     total_reward = 0.0
#     step_count = 0

#     action = agent.select_action(observation, info)

#     while True:
#         next_observation, reward, terminated, truncated, next_info = env.step(action)
#         total_reward += float(reward)
#         step_count += 1
        
#         if terminated or truncated:
#             if hasattr(agent, "update"):
#                 agent.update(
#                     state=observation,
#                     action=action,
#                     reward=float(reward),
#                     next_state=next_observation,
#                     next_action=None, # Действия больше нет
#                     terminated=terminated,
#                     next_info=next_info
#                 )
#             return total_reward, bool(next_info.get("is_win", False)), step_count

#         next_action = agent.select_action(next_observation, next_info)
        
#         if hasattr(agent, "update"):
#             agent.update(
#                 state=observation,
#                 action=action,
#                 reward=float(reward),
#                 next_state=next_observation,
#                 next_action=next_action,
#                 terminated=terminated,
#                 next_info=next_info
#             )

#         observation = next_observation
#         info = next_info
#         action = next_action


# def save_learning_curve(returns: list[float], output_path: Path) -> None:
#     figure = plt.figure(figsize=(8, 4))
#     axes = figure.add_subplot(111)
#     window = min(50, max(1, len(returns) // 10))
#     smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
    
#     axes.plot(range(1, len(returns) + 1), returns, alpha=0.3, color='blue', label='Raw Return')
#     axes.plot(range(window, len(returns) + 1), smoothed, color='red', label=f'MA ({window})')
    
#     axes.set_title("Episode Return during Training")
#     axes.set_xlabel("Episode")
#     axes.set_ylabel("Return")
#     axes.legend()
#     axes.grid(True, alpha=0.3)
#     figure.tight_layout()
#     figure.savefig(output_path)
#     plt.close(figure)

# def save_success_rate_histogram(wins: list[bool], output_path: Path, window_size: int = 100) -> None:
#     """Persist win rate over time as a bar chart."""
#     if len(wins) < window_size:
#         window_size = max(1, len(wins) // 5)
        
#     chunks = [wins[i:i + window_size] for i in range(0, len(wins), window_size)]
#     win_rates = [sum(chunk) / len(chunk) * 100 for chunk in chunks]
#     x_labels = [f"{i*window_size}-{(i+1)*window_size}" for i in range(len(chunks))]

#     figure = plt.figure(figsize=(10, 4))
#     axes = figure.add_subplot(111)
#     axes.bar(x_labels, win_rates, color='green', alpha=0.7)
#     axes.set_title(f"Success Rate per {window_size} Episodes")
#     axes.set_xlabel("Episode Range")
#     axes.set_ylabel("Win Rate (%)")
#     axes.set_ylim(0, 100)
#     plt.xticks(rotation=45)
#     figure.tight_layout()
#     figure.savefig(output_path)
#     plt.close(figure)


# def main() -> None:
#     args = parse_args()
#     configure_logging()
#     base_cfg_path = Path("configs/default.yaml")
#     cfg = load_yaml(base_cfg_path) if base_cfg_path.exists() else {}

#     if args.config != "configs/default.yaml":
#         user_cfg = load_yaml(args.config)
#         for key, value in user_cfg.items():
#             if isinstance(value, dict) and key in cfg:
#                 cfg[key].update(value)
#             else:
#                 cfg[key] = value
    
#     env_cfg_dict = cfg.get("env", {})
#     train_cfg = cfg.get("train", {})
#     paths_cfg = cfg.get("paths", {})

#     if args.seed is not None:
#         env_cfg_dict = {**env_cfg_dict, "seed": args.seed}

#     env_cfg = build_env_config(env_cfg_dict)
#     env = PacmanEnv(config=env_cfg, render_mode=None)

#     episodes = int(args.episodes if args.episodes is not None else train_cfg.get("episodes", 20))
    
#     agent_type = train_cfg.get("agent_type", "random")
#     agent_params = train_cfg.get("agent_params", {})
    
#     AgentClass = AGENT_REGISTRY.get(agent_type)
#     if not AgentClass:
#          raise ValueError(f"Unknown agent type '{agent_type}'. Available: {list(AGENT_REGISTRY.keys())}")
    
#     print(f"Initializing {agent_type} agent with params: {agent_params}")
#     agent = AgentClass(**agent_params)

#     epsilon_end = float(train_cfg.get("epsilon_end", 0.05))
#     epsilon_decay = float(train_cfg.get("epsilon_decay", 0.995))

#     default_output = paths_cfg.get("train_output_dir", "results/train")
#     output_dir = ensure_directory(args.output_dir or default_output)

#     returns: list[float] = []
#     wins: list[bool] = []
#     steps: list[int] = []

#     for episode_idx in range(episodes):
#         episode_seed = int(env_cfg.seed + episode_idx)
        
#         total_reward, did_win, episode_steps = train_episode(env, agent, seed=episode_seed)
        
#         returns.append(total_reward)
#         wins.append(did_win)
#         steps.append(episode_steps)

#         if hasattr(agent, 'epsilon'):
#             agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

#         if (episode_idx + 1) % 100 == 0:
#             print(f"Episode {episode_idx + 1}/{episodes} | Return: {total_reward:.1f} | Won: {did_win}")

#     env.close()

#     mean_return = float(sum(returns) / max(1, len(returns)))
#     win_rate = float(sum(1 for value in wins if value) / max(1, len(wins)))

#     curve_path = output_dir / "learning_curve.png"
#     hist_path = output_dir / "success_rate.png"
#     save_learning_curve(returns, curve_path)
#     save_success_rate_histogram(wins, hist_path, window_size=max(10, episodes // 10))

#     with (output_dir / "episode_returns.csv").open("w", encoding="utf-8") as handle:
#         handle.write("episode,return,win,steps\n")
#         for idx, (ret, win, step_count) in enumerate(zip(returns, wins, steps), start=1):
#             handle.write(f"{idx},{ret:.6f},{int(win)},{step_count}\n")

#     metrics: dict[str, Any] = {
#         "episodes": episodes,
#         "mean_return": mean_return,
#         "win_rate": win_rate,
#         "artifact_learning_curve": str(curve_path),
#         "artifact_success_rate": str(hist_path),
#     }
#     save_json(metrics, output_dir / "train_metrics.json")

#     if hasattr(agent, "save_policy"):
#         policy_path = output_dir / f"{agent_type}_policy.yaml"
#         agent.save_policy(policy_path)
#         metrics["artifact_policy"] = str(policy_path)
    
#     placeholder_model = {
#         "model_type": agent_type,
#         "agent_params": agent_params,
#         "config": cfg,
#         "summary": metrics,
#     }
    
#     save_pickle(placeholder_model, output_dir / "model.pkl")

#     print(f"Training complete. Episodes={episodes}, mean_return={mean_return:.3f}, win_rate={win_rate:.3f}")
#     print(f"Artifacts saved to: {output_dir}")

# if __name__ == "__main__":
#     main()

# import argparse
# import yaml

# import argparse
# from pathlib import Path
# import sys
# from typing import Any

# import matplotlib.pyplot as plt
# import numpy as np

# def _bootstrap_src_path() -> None:
#     """Ensure local src directory is importable without editable install."""
#     project_root = Path(__file__).resolve().parents[1]
#     src_path = project_root / "src"
#     if str(src_path) not in sys.path:
#         sys.path.insert(0, str(src_path))
 
# _bootstrap_src_path()
 
# from pacman_rldp.agents import RandomPolicy
# from pacman_rldp.agents.sarsa import SarsaAgent
# from pacman_rldp.env import PacmanEnv, build_env_config
# from pacman_rldp.logging import configure_logging
# from pacman_rldp.utils import ensure_directory, save_json, save_pickle, load_yaml
 
# AGENT_REGISTRY = {
#     "random": RandomPolicy,
#     "sarsa": SarsaAgent,
# }
 
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Train scaffold for Pacman DP/RL project.")
#     parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
#     parser.add_argument("--output-dir", default=None, help="Override training output directory.")
#     parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes.")
#     parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
#     return parser.parse_args()
 
 
# # def train_episode(env: PacmanEnv, agent: Any, seed: int) -> tuple[float, bool, int]:
# #     observation, info = env.reset(seed=seed)
# #     total_reward = 0.0
# #     step_count = 0
 
# #     action = agent.select_action(observation, info)
 
# #     while True:
# #         next_observation, reward, terminated, truncated, next_info = env.step(action)
# #         total_reward += float(reward)
# #         step_count += 1
 
# #         if terminated or truncated:
# #             if hasattr(agent, "update"):
# #                 # next_action=None сигнализирует о конце эпизода
# #                 agent.update(
# #                     state=observation,
# #                     action=action,
# #                     reward=float(reward),
# #                     next_state=next_observation,
# #                     next_action=None,
# #                     terminated=terminated,
# #                 )
# #             return total_reward, bool(next_info.get("is_win", False)), step_count
 
# #         next_action = agent.select_action(next_observation, next_info)
 
# #         if hasattr(agent, "update"):
# #             agent.update(
# #                 state=observation,
# #                 action=action,
# #                 reward=float(reward),
# #                 next_state=next_observation,
# #                 next_action=next_action,
# #                 terminated=terminated,
# #             )
 
# #         observation = next_observation
# #         info = next_info
# #         action = next_action

# def compute_shaping(obs_before: dict, obs_after: dict, gamma: float) -> float:
#     def min_food_dist(obs) -> float:
#         pac = obs["pacman_position"]
#         food = np.argwhere(obs["food"] == 1)
#         if len(food) == 0:
#             return 0.0
#         return float(np.abs(food - pac).sum(axis=1).min())

#     # Потенциал: чем ближе к еде — тем лучше
#     phi_before = -min_food_dist(obs_before)
#     phi_after  = -min_food_dist(obs_after)
#     return gamma * phi_after - phi_before


# def train_episode(env: PacmanEnv, agent: Any, seed: int) -> tuple[float, bool, int]:
#     observation, info = env.reset(seed=seed)
#     total_reward = 0.0
#     step_count = 0
#     action = agent.select_action(observation, info)

#     while True:
#         next_observation, reward, terminated, truncated, next_info = env.step(action)

#         # Базовая награда + shaping
#         shaping = compute_shaping(observation, next_observation, agent.gamma)
#         shaped_reward = float(reward) + shaping

#         total_reward += float(reward)  # логируем оригинальную награду
#         step_count += 1

#         if terminated or truncated:
#             if hasattr(agent, "update"):
#                 agent.update(
#                     state=observation,
#                     action=action,
#                     reward=shaped_reward,
#                     next_state=next_observation,
#                     next_action=None,
#                     terminated=terminated,
#                 )
#             return total_reward, bool(next_info.get("is_win", False)), step_count

#         next_action = agent.select_action(next_observation, next_info)

#         if hasattr(agent, "update"):
#             agent.update(
#                 state=observation,
#                 action=action,
#                 reward=shaped_reward,
#                 next_state=next_observation,
#                 next_action=next_action,
#                 terminated=terminated,
#             )

#         observation = next_observation
#         info = next_info
#         action = next_action
 
# def save_learning_curve(returns: list[float], output_path: Path) -> None:
#     figure = plt.figure(figsize=(8, 4))
#     axes = figure.add_subplot(111)
#     window = min(50, max(1, len(returns) // 10))
#     smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
 
#     axes.plot(range(1, len(returns) + 1), returns, alpha=0.3, color='blue', label='Raw Return')
#     axes.plot(range(window, len(returns) + 1), smoothed, color='red', label=f'MA ({window})')
 
#     axes.set_title("Episode Return during Training")
#     axes.set_xlabel("Episode")
#     axes.set_ylabel("Return")
#     axes.legend()
#     axes.grid(True, alpha=0.3)
#     figure.tight_layout()
#     figure.savefig(output_path)
#     plt.close(figure)
 
# def save_success_rate_histogram(wins: list[bool], output_path: Path, window_size: int = 100) -> None:
#     """Persist win rate over time as a bar chart."""
#     if len(wins) < window_size:
#         window_size = max(1, len(wins) // 5)
 
#     chunks = [wins[i:i + window_size] for i in range(0, len(wins), window_size)]
#     win_rates = [sum(chunk) / len(chunk) * 100 for chunk in chunks]
#     x_labels = [f"{i*window_size}-{(i+1)*window_size}" for i in range(len(chunks))]
 
#     figure = plt.figure(figsize=(10, 4))
#     axes = figure.add_subplot(111)
#     axes.bar(x_labels, win_rates, color='green', alpha=0.7)
#     axes.set_title(f"Success Rate per {window_size} Episodes")
#     axes.set_xlabel("Episode Range")
#     axes.set_ylabel("Win Rate (%)")
#     axes.set_ylim(0, 100)
#     plt.xticks(rotation=45)
#     figure.tight_layout()
#     figure.savefig(output_path)
#     plt.close(figure)
 
 
# def main() -> None:
#     args = parse_args()
#     configure_logging()
#     base_cfg_path = Path("configs/default.yaml")
#     cfg = load_yaml(base_cfg_path) if base_cfg_path.exists() else {}
 
#     if args.config != "configs/default.yaml":
#         user_cfg = load_yaml(args.config)
#         for key, value in user_cfg.items():
#             if isinstance(value, dict) and key in cfg:
#                 cfg[key].update(value)
#             else:
#                 cfg[key] = value
 
#     env_cfg_dict = cfg.get("env", {})
#     train_cfg = cfg.get("train", {})
#     paths_cfg = cfg.get("paths", {})
 
#     if args.seed is not None:
#         env_cfg_dict = {**env_cfg_dict, "seed": args.seed}
 
#     env_cfg = build_env_config(env_cfg_dict)
#     env = PacmanEnv(config=env_cfg, render_mode=None)
 
#     episodes = int(args.episodes if args.episodes is not None else train_cfg.get("episodes", 20))
 
#     agent_type = train_cfg.get("agent_type", "random")
#     agent_params = train_cfg.get("agent_params", {})
 
#     AgentClass = AGENT_REGISTRY.get(agent_type)
#     if not AgentClass:
#         raise ValueError(f"Unknown agent type '{agent_type}'. Available: {list(AGENT_REGISTRY.keys())}")
 
#     print(f"Initializing {agent_type} agent with params: {agent_params}")
#     agent = AgentClass(**agent_params)
 
#     # Инициализируем отдельный seed агента для воспроизводимости,
#     # независимо от глобального seed среды
#     if hasattr(agent, "seed"):
#         agent.seed(env_cfg.seed)
 
#     epsilon_end = float(train_cfg.get("epsilon_end", 0.05))
#     epsilon_decay = float(train_cfg.get("epsilon_decay", 0.9995))
 
#     default_output = paths_cfg.get("train_output_dir", "results/train")
#     output_dir = ensure_directory(args.output_dir or default_output)
 
#     returns: list[float] = []
#     wins: list[bool] = []
#     steps: list[int] = []
 
#     for episode_idx in range(episodes):
#         episode_seed = int(env_cfg.seed + episode_idx)
 
#         total_reward, did_win, episode_steps = train_episode(env, agent, seed=episode_seed)
 
#         returns.append(total_reward)
#         wins.append(did_win)
#         steps.append(episode_steps)
 
#         if hasattr(agent, 'epsilon'):
#             agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)
 
#         if (episode_idx + 1) % 100 == 0:
#             print(f"Episode {episode_idx + 1}/{episodes} | Return: {total_reward:.1f} | Won: {did_win}")
#             print(f"Уникальных состояний в Q-таблице: {len(agent.q_table)}")
#             print(f"Макс Q-значение: {max(v.max() for v in agent.q_table.values()):.3f}")
#             print(f"Мин Q-значение: {min(v.min() for v in agent.q_table.values()):.3f}")
 
#     env.close()
 
#     # Метрики считаем по последним 20% эпизодов — агент уже обучен,
#     # ранние эпизоды с высоким epsilon не искажают итоговую оценку
#     eval_window = max(1, episodes // 5)
#     mean_return = float(sum(returns[-eval_window:]) / eval_window)
#     win_rate = float(sum(1 for v in wins[-eval_window:] if v) / eval_window)
 
#     curve_path = output_dir / "learning_curve.png"
#     hist_path = output_dir / "success_rate.png"
#     save_learning_curve(returns, curve_path)
#     save_success_rate_histogram(wins, hist_path, window_size=max(10, episodes // 10))
 
#     with (output_dir / "episode_returns.csv").open("w", encoding="utf-8") as handle:
#         handle.write("episode,return,win,steps\n")
#         for idx, (ret, win, step_count) in enumerate(zip(returns, wins, steps), start=1):
#             handle.write(f"{idx},{ret:.6f},{int(win)},{step_count}\n")
 
#     metrics: dict[str, Any] = {
#         "episodes": episodes,
#         "mean_return": mean_return,
#         "win_rate": win_rate,
#         "eval_window": eval_window,
#         "artifact_learning_curve": str(curve_path),
#         "artifact_success_rate": str(hist_path),
#     }
#     save_json(metrics, output_dir / "train_metrics.json")
 
#     if hasattr(agent, "save_policy"):
#         policy_path = output_dir / f"{agent_type}_policy.yaml"
#         agent.save_policy(policy_path)
#         metrics["artifact_policy"] = str(policy_path)
 
#     placeholder_model = {
#         "model_type": agent_type,
#         "agent_params": agent_params,
#         "config": cfg,
#         "summary": metrics,
#     }
 
#     save_pickle(placeholder_model, output_dir / "model.pkl")
 
#     print(f"Training complete. Episodes={episodes}, mean_return={mean_return:.3f}, win_rate={win_rate:.3f}")
#     print(f"Artifacts saved to: {output_dir}")
 
# if __name__ == "__main__":
#     main()

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _bootstrap_src_path() -> None:
    """Ensure local src directory is importable without editable install."""
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from pacman_rldp.agents import RandomPolicy
from pacman_rldp.agents.sarsa import SarsaAgent
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.logging import configure_logging
from pacman_rldp.utils import ensure_directory, save_json, save_pickle, load_yaml

AGENT_REGISTRY = {
    "random": RandomPolicy,
    "sarsa": SarsaAgent,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scaffold for Pacman DP/RL project.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override training output directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def compute_shaping(obs_before: dict, obs_after: dict, gamma: float) -> float:
    # Ветка BITMASK_DISTANCE_BUCKETS: используем готовый бакет расстояния до еды.
    # Меньший бакет = ближе к еде = лучший потенциал.
    if "nearest_food_bucket" in obs_before:
        bucket_before = int(obs_before["nearest_food_bucket"][0])
        bucket_after  = int(obs_after["nearest_food_bucket"][0])
        # sentinel -1 означает еды нет — потенциал 0
        phi_before = 0.0 if bucket_before < 0 else float(-bucket_before)
        phi_after  = 0.0 if bucket_after  < 0 else float(-bucket_after)
        return gamma * phi_after - phi_before

    # Ветка RAW: считаем манхэттенское расстояние до ближайшей еды напрямую
    def min_food_dist(obs) -> float:
        pac  = obs["pacman_position"]
        food = np.argwhere(obs["food"] == 1)
        if len(food) == 0:
            return 0.0
        return float(np.abs(food - pac).sum(axis=1).min())

    phi_before = -min_food_dist(obs_before)
    phi_after  = -min_food_dist(obs_after)
    return gamma * phi_after - phi_before


def train_episode(env: PacmanEnv, agent: Any, seed: int) -> tuple[float, bool, int]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    step_count   = 0
    action       = agent.select_action(observation, info)

    while True:
        next_observation, reward, terminated, truncated, next_info = env.step(action)

        shaping      = compute_shaping(observation, next_observation, agent.gamma)
        shaped_reward = float(reward) + shaping
        total_reward += float(reward)
        step_count   += 1

        if terminated or truncated:
            if hasattr(agent, "update"):
                agent.update(
                    state=observation,
                    action=action,
                    reward=shaped_reward,
                    next_state=next_observation,
                    next_action=None,
                    terminated=terminated,
                )
            return total_reward, bool(next_info.get("is_win", False)), step_count

        next_action = agent.select_action(next_observation, next_info)

        if hasattr(agent, "update"):
            agent.update(
                state=observation,
                action=action,
                reward=shaped_reward,
                next_state=next_observation,
                next_action=next_action,
                terminated=terminated,
            )

        observation = next_observation
        info        = next_info
        action      = next_action


def save_learning_curve(returns: list[float], output_path: Path) -> None:
    figure = plt.figure(figsize=(8, 4))
    axes   = figure.add_subplot(111)
    window = min(50, max(1, len(returns) // 10))
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")

    axes.plot(range(1, len(returns) + 1), returns, alpha=0.3, color="blue", label="Raw Return")
    axes.plot(range(window, len(returns) + 1), smoothed, color="red", label=f"MA ({window})")
    axes.set_title("Episode Return during Training")
    axes.set_xlabel("Episode")
    axes.set_ylabel("Return")
    axes.legend()
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_success_rate_histogram(wins: list[bool], output_path: Path, window_size: int = 100) -> None:
    """Persist win rate over time as a bar chart."""
    if len(wins) < window_size:
        window_size = max(1, len(wins) // 5)

    chunks    = [wins[i:i + window_size] for i in range(0, len(wins), window_size)]
    win_rates = [sum(chunk) / len(chunk) * 100 for chunk in chunks]
    x_labels  = [f"{i * window_size}-{(i + 1) * window_size}" for i in range(len(chunks))]

    figure = plt.figure(figsize=(10, 4))
    axes   = figure.add_subplot(111)
    axes.bar(x_labels, win_rates, color="green", alpha=0.7)
    axes.set_title(f"Success Rate per {window_size} Episodes")
    axes.set_xlabel("Episode Range")
    axes.set_ylabel("Win Rate (%)")
    axes.set_ylim(0, 100)
    plt.xticks(rotation=45)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _log_episode(episode_idx: int, episodes: int, total_reward: float, did_win: bool, agent: Any) -> None:
    print(f"Episode {episode_idx + 1}/{episodes} | Return: {total_reward:.1f} | Won: {did_win}")
    if hasattr(agent, "q_table") and agent.q_table:
        q_vals = list(agent.q_table.values())
        print(f"  States: {len(q_vals)} | "
              f"Max Q: {max(v.max() for v in q_vals):.3f} | "
              f"Min Q: {min(v.min() for v in q_vals):.3f}")
    if hasattr(agent, "epsilon"):
        print(f"  Epsilon: {agent.epsilon:.4f}")


def main() -> None:
    args = parse_args()
    configure_logging()

    base_cfg_path = Path("configs/default.yaml")
    cfg = load_yaml(base_cfg_path) if base_cfg_path.exists() else {}

    if args.config != "configs/default.yaml":
        user_cfg = load_yaml(args.config)
        for key, value in user_cfg.items():
            if isinstance(value, dict) and key in cfg:
                cfg[key].update(value)
            else:
                cfg[key] = value

    env_cfg_dict = cfg.get("env", {})
    train_cfg    = cfg.get("train", {})
    paths_cfg    = cfg.get("paths", {})

    if args.seed is not None:
        env_cfg_dict = {**env_cfg_dict, "seed": args.seed}

    env_cfg = build_env_config(env_cfg_dict)
    env     = PacmanEnv(config=env_cfg, render_mode=None)

    episodes = int(args.episodes if args.episodes is not None else train_cfg.get("episodes", 20))

    agent_type   = train_cfg.get("agent_type", "random")
    agent_params = train_cfg.get("agent_params", {})

    AgentClass = AGENT_REGISTRY.get(agent_type)
    if not AgentClass:
        raise ValueError(f"Unknown agent type '{agent_type}'. Available: {list(AGENT_REGISTRY.keys())}")

    print(f"Initializing {agent_type} agent with params: {agent_params}")
    agent = AgentClass(**agent_params)

    if hasattr(agent, "seed"):
        agent.seed(env_cfg.seed)

    epsilon_end   = float(train_cfg.get("epsilon_end", 0.05))
    epsilon_decay = float(train_cfg.get("epsilon_decay", 0.9995))

    default_output = paths_cfg.get("train_output_dir", "results/train")
    output_dir     = ensure_directory(args.output_dir or default_output)

    returns: list[float] = []
    wins:    list[bool]  = []
    steps:   list[int]   = []

    for episode_idx in range(episodes):
        episode_seed = int(env_cfg.seed + episode_idx)

        total_reward, did_win, episode_steps = train_episode(env, agent, seed=episode_seed)

        returns.append(total_reward)
        wins.append(did_win)
        steps.append(episode_steps)

        if hasattr(agent, "epsilon"):
            agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

        if (episode_idx + 1) % 100 == 0:
            _log_episode(episode_idx, episodes, total_reward, did_win, agent)

    env.close()

    eval_window = max(1, episodes // 5)
    mean_return = float(sum(returns[-eval_window:]) / eval_window)
    win_rate    = float(sum(1 for v in wins[-eval_window:] if v) / eval_window)

    curve_path = output_dir / "learning_curve.png"
    hist_path  = output_dir / "success_rate.png"
    save_learning_curve(returns, curve_path)
    save_success_rate_histogram(wins, hist_path, window_size=max(10, episodes // 10))

    with (output_dir / "episode_returns.csv").open("w", encoding="utf-8") as handle:
        handle.write("episode,return,win,steps\n")
        for idx, (ret, win, step_count) in enumerate(zip(returns, wins, steps), start=1):
            handle.write(f"{idx},{ret:.6f},{int(win)},{step_count}\n")

    metrics: dict[str, Any] = {
        "episodes":               episodes,
        "mean_return":            mean_return,
        "win_rate":               win_rate,
        "eval_window":            eval_window,
        "artifact_learning_curve": str(curve_path),
        "artifact_success_rate":  str(hist_path),
    }
    save_json(metrics, output_dir / "train_metrics.json")

    if hasattr(agent, "save_policy"):
        policy_path = output_dir / f"{agent_type}_policy.yaml"
        agent.save_policy(policy_path)
        metrics["artifact_policy"] = str(policy_path)

    save_pickle(
        {"model_type": agent_type, "agent_params": agent_params, "config": cfg, "summary": metrics},
        output_dir / "model.pkl",
    )

    print(f"Training complete. Episodes={episodes}, mean_return={mean_return:.3f}, win_rate={win_rate:.3f}")
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()