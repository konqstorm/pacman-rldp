# import random
# import pickle
# import argparse
# import time
# import sys
# from pathlib import Path

# import numpy as np


# def _bootstrap_src_path() -> None:
#     project_root = Path(__file__).resolve().parents[1]
#     src_path = project_root / "src"
#     if str(src_path) not in sys.path:
#         sys.path.insert(0, str(src_path))


# _bootstrap_src_path()

# from pacman_rldp.env import PacmanEnv, build_env_config
# from pacman_rldp.utils import load_yaml
# from pacman_rldp.third_party.bk.util import Counter, manhattanDistance
# from pacman_rldp.third_party.bk.game import Actions
# from pacman_rldp.third_party.bk import graphicsUtils

# try:
#     from PIL import Image, ImageGrab
#     _PIL_AVAILABLE = True
# except ImportError:
#     _PIL_AVAILABLE = False


# # ---------------------------------------------------------------------------
# # Извлечение признаков
# # ---------------------------------------------------------------------------

# class FeatureExtractor:
#     """
#     Признаки для Approximate SARSA.
#     Все float-признаки нормированы в [0, 1].

#     Список признаков:
#       bias            — свободный член, всегда 1.0
#       hit-wall        — действие ведёт в стену
#       scared          — хоть один призрак испуган
#       eats-food       — в следующей клетке есть еда
#       closest-food    — BFS-расстояние до ближайшей еды, нормированное
#       food-nearby     — плотность еды в 3×3 окрестности / 9
#       ghost-distance  — манхэттен до ближайшего призрака, нормированное
#       danger          — призрак в 1 клетке от следующей позиции
#       capsule-distance— манхэттен до ближайшей капсулы, нормированное
#       stop            — действие STOP
#     """

#     def get_features(self, state, action: str) -> Counter:
#         features = Counter()
#         features["bias"] = 1.0

#         pac_pos  = state.getPacmanPosition()
#         food     = state.getFood()
#         walls    = state.getWalls()
#         ghosts   = state.getGhostStates()
#         capsules = state.getCapsules()
#         norm     = float(walls.width + walls.height)

#         dx, dy   = Actions.directionToVector(action)
#         next_x   = int(pac_pos[0] + dx)
#         next_y   = int(pac_pos[1] + dy)

#         # Стена
#         if (next_x < 0 or next_x >= walls.width
#                 or next_y < 0 or next_y >= walls.height
#                 or walls[next_x][next_y]):
#             features["hit-wall"] = 1.0
#             return features

#         next_pos = (next_x, next_y)

#         # Испуг призраков
#         if any(gs.scaredTimer > 0 for gs in ghosts if gs.getPosition() is not None):
#             features["scared"] = 1.0

#         # Еда в следующей клетке
#         if food[next_x][next_y]:
#             features["eats-food"] = 1.0

#         # BFS до ближайшей еды
#         dist = self._bfs_food(next_pos, food, walls)
#         if dist is not None:
#             features["closest-food"] = float(dist) / norm

#         # Плотность еды в 3×3
#         food_count = sum(
#             1
#             for dx2 in [-1, 0, 1]
#             for dy2 in [-1, 0, 1]
#             if (0 <= next_x + dx2 < walls.width
#                 and 0 <= next_y + dy2 < walls.height
#                 and food[next_x + dx2][next_y + dy2])
#         )
#         features["food-nearby"] = food_count / 9.0

#         # Расстояние до призраков
#         active = [gs for gs in ghosts if gs.getPosition() is not None]
#         if active:
#             min_dist = min(manhattanDistance(next_pos, gs.getPosition()) for gs in active)
#             features["ghost-distance"] = float(min_dist) / norm
#             if min_dist <= 1:
#                 features["danger"] = 1.0

#         # Расстояние до капсулы
#         if capsules:
#             cap_dist = min(manhattanDistance(next_pos, c) for c in capsules)
#             features["capsule-distance"] = float(cap_dist) / norm

#         # STOP
#         if action == "Stop":
#             features["stop"] = 1.0

#         return features

#     def _bfs_food(self, start: tuple, food, walls) -> int | None:
#         fringe   = [(start[0], start[1], 0)]
#         expanded = set()
#         while fringe:
#             x, y, d = fringe.pop(0)
#             if (x, y) in expanded:
#                 continue
#             expanded.add((x, y))
#             if food[x][y]:
#                 return d
#             for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
#                 nx, ny = x + dx, y + dy
#                 if (0 <= nx < walls.width and 0 <= ny < walls.height
#                         and not walls[nx][ny]):
#                     fringe.append((nx, ny, d + 1))
#         return None


# # ---------------------------------------------------------------------------
# # Approximate SARSA(λ) агент
# # ---------------------------------------------------------------------------

# class SarsaAgent:
#     """
#     Approximate SARSA(λ).

#     Q(s,a) = w · f(s,a)  — линейная аппроксимация

#     Обновление весов:
#       δ  = r + γ·Q(s',a') - Q(s,a)
#       e  += f(s,a)                    — накопить след
#       w  += α·δ·e                     — обновить веса
#       e  *= γ·λ                       — затухание следов
#     """

#     ACTION_MAP = ["North", "South", "East", "West", "Stop"]

#     def __init__(self, action_space_size: int = 5, alpha: float = 0.01,
#                  gamma: float = 0.9, epsilon: float = 1.0, lam: float = 0.8):
#         self.alpha             = alpha
#         self.gamma             = gamma
#         self.epsilon           = epsilon
#         self.lam               = lam
#         self.action_space_size = action_space_size
#         self.extractor         = FeatureExtractor()
#         self.weights           = Counter()
#         self.traces            = Counter()

#     def _q_value(self, state, action_id: int) -> float:
#         features = self.extractor.get_features(state, self.ACTION_MAP[action_id])
#         return float(features * self.weights)

#     def choose_action(self, state, legal_actions: list[int]) -> int:
#         # Убираем STOP
#         legal = [a for a in legal_actions if a != 4]
#         if not legal:
#             legal = legal_actions

#         if random.random() < self.epsilon:
#             return random.choice(legal)

#         q_values     = {a: self._q_value(state, a) for a in legal}
#         max_q        = max(q_values.values())
#         best_actions = [a for a, q in q_values.items() if q == max_q]
#         return random.choice(best_actions)

#     def learn(self, state, action: int, reward: float,
#               next_state, next_action: int | None, done: bool) -> None:
#         current_q = self._q_value(state, action)

#         if done or next_action is None:
#             next_q = 0.0
#         else:
#             next_q = self._q_value(next_state, next_action)

#         delta    = reward + self.gamma * next_q - current_q
#         features = self.extractor.get_features(state, self.ACTION_MAP[action])

#         # Накопить след для текущих признаков
#         for feat, val in features.items():
#             self.traces[feat] += val

#         # Обновить веса и затухание следов
#         for feat, trace_val in list(self.traces.items()):
#             self.weights[feat] += self.alpha * delta * trace_val
#             self.traces[feat]  *= self.gamma * self.lam
#             if abs(self.traces[feat]) < 1e-6:
#                 del self.traces[feat]

#         if done:
#             self.traces.clear()

#     def save(self, path: str | Path) -> None:
#         with open(path, "wb") as f:
#             pickle.dump({
#                 "weights": dict(self.weights),
#                 "alpha":   self.alpha,
#                 "gamma":   self.gamma,
#                 "lam":     self.lam,
#             }, f)
#         print(f"Model saved to {path}")

#     def load(self, path: str | Path) -> None:
#         with open(path, "rb") as f:
#             data = pickle.load(f)
#         self.alpha   = data.get("alpha", self.alpha)
#         self.gamma   = data.get("gamma", self.gamma)
#         self.lam     = data.get("lam",   self.lam)
#         self.weights = Counter(data["weights"])
#         print(f"Model loaded. Weights: {dict(self.weights)}")


# # ---------------------------------------------------------------------------
# # Reward shaping
# # ---------------------------------------------------------------------------

# def compute_shaping(state, next_state, gamma: float) -> float:
#     food = next_state.getFood().asList()
#     if not food:
#         return 0.0
#     pac_before = state.getPacmanPosition()
#     pac_after  = next_state.getPacmanPosition()
#     d_before   = min(manhattanDistance(pac_before, f) for f in food)
#     d_after    = min(manhattanDistance(pac_after,  f) for f in food)
#     return gamma * (-d_after) - (-d_before)


# # ---------------------------------------------------------------------------
# # Цикл обучения
# # ---------------------------------------------------------------------------

# def train(env: PacmanEnv, agent: SarsaAgent, episodes: int,
#           epsilon_end: float = 0.05, epsilon_decay: float = 0.9995) -> list[float]:
#     print(f"Starting training for {episodes} episodes...")
#     rewards_history: list[float] = []
#     wins_history:    list[bool]  = []

#     for i in range(episodes):

#         obs, info = env.reset(env.seed + i)
#         state        = env.runtime_state
#         total_reward = 0.0
#         done         = False

#         agent.traces.clear()

#         legal_actions = info.get("legal_action_ids", [])
#         action        = agent.choose_action(state, legal_actions)

#         while not done:
#             _, reward, terminated, truncated, info = env.step(action)
#             done         = terminated or truncated
#             next_state   = env.runtime_state

#             shaping       = compute_shaping(state, next_state, agent.gamma)
#             shaped_reward = reward + shaping
#             total_reward += reward

#             if done:
#                 agent.learn(state, action, shaped_reward, next_state, None, done)
#             else:
#                 next_legal  = info.get("legal_action_ids", [])
#                 next_action = agent.choose_action(next_state, next_legal)
#                 agent.learn(state, action, shaped_reward, next_state, next_action, done)
#                 action = next_action

#             state = next_state

#         agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)
#         rewards_history.append(total_reward)
#         wins_history.append(bool(info.get("is_win", False)))

#         if (i + 1) % 100 == 0:
#             avg      = np.mean(rewards_history[-100:])
#             win_rate = np.mean(wins_history[-100:]) * 100
#             print(f"Episode {i+1}/{episodes} | "
#                   f"Avg(100): {avg:.1f} | "
#                   f"WinRate(100): {win_rate:.1f}% | "
#                   f"Epsilon: {agent.epsilon:.4f} | "
#                   f"Weights: {dict(agent.weights)}")

#     return rewards_history


# # ---------------------------------------------------------------------------
# # GIF запись
# # ---------------------------------------------------------------------------

# def capture_frame():
#     if not _PIL_AVAILABLE:
#         return None
#     canvas = graphicsUtils._canvas
#     root   = graphicsUtils._root_window
#     if canvas is None or root is None:
#         return None
#     root.update_idletasks()
#     root.update()
#     try:
#         x0, y0 = int(canvas.winfo_rootx()), int(canvas.winfo_rooty())
#         w, h   = int(canvas.winfo_width()), int(canvas.winfo_height())
#         if w <= 1 or h <= 1:
#             return None
#         return ImageGrab.grab(bbox=(x0, y0, x0 + w, y0 + h))
#     except Exception:
#         return None


# def save_gif(frames: list, output_path: Path, frame_time: float = 0.1) -> None:
#     if not frames or not _PIL_AVAILABLE:
#         return
#     duration_ms = max(20, int(frame_time * 1000))
#     frames[0].save(output_path, save_all=True, append_images=frames[1:],
#                    optimize=False, duration=duration_ms, loop=0)
#     print(f"GIF saved to {output_path}")


# # ---------------------------------------------------------------------------
# # main
# # ---------------------------------------------------------------------------

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Approximate SARSA(λ) for Pacman")
#     parser.add_argument("--config",        type=str,   default="configs/default.yaml")
#     parser.add_argument("--episodes",      type=int,   default=3000)
#     parser.add_argument("--train",         action="store_true")
#     parser.add_argument("--eval",          action="store_true")
#     parser.add_argument("--model",         type=str,   default="results/sarsa_approx.pkl")
#     parser.add_argument("--render",        action="store_true")
#     parser.add_argument("--seed",          type=int,   default=None)
#     parser.add_argument("--record",        action="store_true")
#     parser.add_argument("--record-path",   type=str,   default="results/replay.gif")
#     parser.add_argument("--lam",           type=float, default=0.8)
#     parser.add_argument("--alpha",         type=float, default=0.01)
#     parser.add_argument("--gamma",         type=float, default=0.9)
#     parser.add_argument("--epsilon-end",   type=float, default=0.05)
#     parser.add_argument("--epsilon-decay", type=float, default=0.9995)
#     args = parser.parse_args()

#     config_dict = load_yaml(args.config)
#     env_config  = build_env_config(config_dict.get("env", config_dict))

#     print(f"Reward config: lose={env_config.reward.lose}, food={env_config.reward.food}")

#     if args.seed is not None:
#         env_config.seed = args.seed

#     env   = PacmanEnv(env_config)
#     agent = SarsaAgent(
#         alpha   = args.alpha,
#         gamma   = args.gamma,
#         epsilon = 1.0,
#         lam     = args.lam,
#     )

#     if args.train:
#         train(env, agent, args.episodes,
#               epsilon_end   = args.epsilon_end,
#               epsilon_decay = args.epsilon_decay)
#         Path("results").mkdir(exist_ok=True)
#         agent.save(args.model)

#     if args.eval:
#         if not Path(args.model).exists():
#             print(f"Model file {args.model} not found!")
#             return
#         agent.load(args.model)
#         agent.epsilon = 0.0

#         from dataclasses import replace
#         render_cfg = replace(env_config, render_mode="human")
#         render_env = PacmanEnv(render_cfg, render_mode="human")

#         gif_frames: list = []
#         obs, info = render_env.reset(seed=env_config.seed)
#         state     = render_env.runtime_state
#         done      = False
#         action    = agent.choose_action(state, info.get("legal_action_ids", []))

#         if args.record:
#             frame = capture_frame()
#             if frame:
#                 gif_frames.append(frame)

#         while not done:
#             _, _, terminated, truncated, info = render_env.step(action)
#             state = render_env.runtime_state
#             done  = terminated or truncated

#             if not done:
#                 action = agent.choose_action(state, info.get("legal_action_ids", []))

#             if args.render or args.record:
#                 render_env.render()
#                 if args.record:
#                     frame = capture_frame()
#                     if frame:
#                         gif_frames.append(frame)
#                 else:
#                     time.sleep(0.05)

#         render_env.close()
#         print("Evaluation finished.")

#         if args.record and gif_frames:
#             output_path = Path(args.record_path)
#             output_path.parent.mkdir(exist_ok=True)
#             save_gif(gif_frames, output_path, env_config.frame_time)

#     env.close()


# if __name__ == "__main__":
#     main()

import random
import pickle
import argparse
import time
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.utils import load_yaml
from pacman_rldp.third_party.bk.util import Counter, manhattanDistance
from pacman_rldp.third_party.bk.game import Actions
from pacman_rldp.third_party.bk import graphicsUtils

try:
    from PIL import Image, ImageGrab
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Извлечение признаков
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Признаки для Approximate SARSA.

    Список признаков:
      bias            — свободный член, всегда 1.0
      hit-wall        — действие ведёт в стену
      scared          — хоть один призрак испуган
      eats-food       — в следующей клетке есть еда
      closest-food    — BFS-расстояние до ближайшей еды, нормированное
      food-nearby     — плотность еды в 3×3 окрестности / 9
      ghost-distance  — манхэттен до ближайшего призрака, нормированное
      danger          — призрак в 1 клетке от следующей позиции
      capsule-distance— манхэттен до ближайшей капсулы, нормированное
      stop            — действие STOP
    """

    def get_features(self, state, action: str) -> Counter:
        features = Counter()
        features["bias"] = 1.0

        pac_pos  = state.getPacmanPosition()
        food     = state.getFood()
        walls    = state.getWalls()
        ghosts   = state.getGhostStates()
        capsules = state.getCapsules()
        norm     = float(walls.width + walls.height)

        dx, dy = Actions.directionToVector(action)
        next_x = int(pac_pos[0] + dx)
        next_y = int(pac_pos[1] + dy)

        if (next_x < 0 or next_x >= walls.width
                or next_y < 0 or next_y >= walls.height
                or walls[next_x][next_y]):
            features["hit-wall"] = 1.0
            return features

        next_pos = (next_x, next_y)

        if any(gs.scaredTimer > 0 for gs in ghosts if gs.getPosition() is not None):
            features["scared"] = 1.0

        if food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self._bfs_food(next_pos, food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / norm

        food_count = sum(
            1
            for dx2 in [-1, 0, 1]
            for dy2 in [-1, 0, 1]
            if (0 <= next_x + dx2 < walls.width
                and 0 <= next_y + dy2 < walls.height
                and food[next_x + dx2][next_y + dy2])
        )
        features["food-nearby"] = food_count / 9.0

        active = [gs for gs in ghosts if gs.getPosition() is not None]
        if active:
            min_dist = min(manhattanDistance(next_pos, gs.getPosition()) for gs in active)
            features["ghost-distance"] = float(min_dist) / norm
            if min_dist <= 1:
                features["danger"] = 1.0

        if capsules:
            cap_dist = min(manhattanDistance(next_pos, c) for c in capsules)
            features["capsule-distance"] = float(cap_dist) / norm

        if action == "Stop":
            features["stop"] = 1.0

        return features

    def _bfs_food(self, start: tuple, food, walls) -> int | None:
        fringe   = [(start[0], start[1], 0)]
        expanded = set()
        while fringe:
            x, y, d = fringe.pop(0)
            if (x, y) in expanded:
                continue
            expanded.add((x, y))
            if food[x][y]:
                return d
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < walls.width and 0 <= ny < walls.height
                        and not walls[nx][ny]):
                    fringe.append((nx, ny, d + 1))
        return None


# ---------------------------------------------------------------------------
# Approximate SARSA(λ) агент
# ---------------------------------------------------------------------------

class SarsaAgent:
    """
    Approximate SARSA(λ).
    Q(s,a) = w · f(s,a)

    Обновление:
      δ = r + γ·Q(s',a') - Q(s,a)
      e += f(s,a)
      w += α·δ·e
      e *= γ·λ
    """

    ACTION_MAP = ["North", "South", "East", "West", "Stop"]

    def __init__(self, action_space_size: int = 5, alpha: float = 0.01,
                 gamma: float = 0.9, epsilon: float = 1.0, lam: float = 0.8):
        self.alpha             = alpha
        self.gamma             = gamma
        self.epsilon           = epsilon
        self.lam               = lam
        self.action_space_size = action_space_size
        self.extractor         = FeatureExtractor()
        self.weights           = Counter()
        self.traces            = Counter()

    def _q_value(self, state, action_id: int) -> float:
        features = self.extractor.get_features(state, self.ACTION_MAP[action_id])
        return float(features * self.weights)

    def choose_action(self, state, legal_actions: list[int]) -> int:
        legal = [a for a in legal_actions if a != 4]
        if not legal:
            legal = legal_actions

        if random.random() < self.epsilon:
            return random.choice(legal)

        q_values     = {a: self._q_value(state, a) for a in legal}
        max_q        = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action: int, reward: float,
              next_state, next_action: int | None, done: bool) -> None:
        current_q = self._q_value(state, action)
        next_q    = 0.0 if (done or next_action is None) else self._q_value(next_state, next_action)
        delta     = reward + self.gamma * next_q - current_q
        features  = self.extractor.get_features(state, self.ACTION_MAP[action])

        for feat, val in features.items():
            self.traces[feat] += val

        for feat, trace_val in list(self.traces.items()):
            self.weights[feat] += self.alpha * delta * trace_val
            self.traces[feat]  *= self.gamma * self.lam
            if abs(self.traces[feat]) < 1e-6:
                del self.traces[feat]

        if done:
            self.traces.clear()

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "weights": dict(self.weights),
                "alpha":   self.alpha,
                "gamma":   self.gamma,
                "lam":     self.lam,
            }, f)
        print(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.alpha   = data.get("alpha", self.alpha)
        self.gamma   = data.get("gamma", self.gamma)
        self.lam     = data.get("lam",   self.lam)
        self.weights = Counter(data["weights"])
        print(f"Model loaded. Weights: {dict(self.weights)}")


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def compute_shaping(state, next_state, gamma: float) -> float:
    food = next_state.getFood().asList()
    if not food:
        return 0.0
    pac_before = state.getPacmanPosition()
    pac_after  = next_state.getPacmanPosition()
    d_before   = min(manhattanDistance(pac_before, f) for f in food)
    d_after    = min(manhattanDistance(pac_after,  f) for f in food)
    return gamma * (-d_after) - (-d_before)


# ---------------------------------------------------------------------------
# График обучения
# ---------------------------------------------------------------------------

def save_learning_curve(rewards: list[float], output_path: Path,
                        window: int = 100) -> None:
    """
    График: сырая награда (прозрачный) + скользящее среднее (жирный).
    По оси X — номер эпизода, по Y — суммарная награда за эпизод.
    """
    episodes = list(range(1, len(rewards) + 1))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smooth_x = list(range(window, len(rewards) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.25, color="steelblue", linewidth=0.8,
            label="Награда за эпизод")
    ax.plot(smooth_x, smoothed, color="steelblue", linewidth=2,
            label=f"Скользящее среднее (окно={window})")

    ax.set_title("Кривая обучения Approximate SARSA(λ)")
    ax.set_xlabel("Эпизод")
    ax.set_ylabel("Суммарная награда")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved to {output_path}")


def train(env: PacmanEnv, agent: SarsaAgent, episodes: int,
          base_seed: int, output_dir: Path,
          epsilon_end: float = 0.05,
          epsilon_decay: float = 0.9995) -> list[float]:
    print(f"Starting training for {episodes} episodes...")
    rewards_history: list[float] = []
    wins_history:    list[bool]  = []

    for i in range(episodes):
        episode_seed = base_seed + i
        obs, info    = env.reset(seed=episode_seed)
        state        = env.runtime_state
        total_reward = 0.0
        done         = False

        agent.traces.clear()

        legal_actions = info.get("legal_action_ids", [])
        action        = agent.choose_action(state, legal_actions)

        while not done:
            _, reward, terminated, truncated, info = env.step(action)
            done         = terminated or truncated
            next_state   = env.runtime_state

            shaping       = compute_shaping(state, next_state, agent.gamma)
            shaped_reward = reward + shaping
            total_reward += reward

            if done:
                agent.learn(state, action, shaped_reward, next_state, None, done)
            else:
                next_legal  = info.get("legal_action_ids", [])
                next_action = agent.choose_action(next_state, next_legal)
                agent.learn(state, action, shaped_reward, next_state, next_action, done)
                action = next_action

            state = next_state

        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        wins_history.append(bool(info.get("is_win", False)))

        if (i + 1) % 100 == 0:
            avg      = np.mean(rewards_history[-100:])
            win_rate = np.mean(wins_history[-100:]) * 100
            print(f"Episode {i+1}/{episodes} | "
                  f"Avg(100): {avg:.1f} | "
                  f"WinRate(100): {win_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Weights: {dict(agent.weights)}")

    # Сохранить график после обучения
    curve_path = output_dir / "learning_curve.png"
    window     = min(100, max(10, episodes // 50))
    save_learning_curve(rewards_history, curve_path, window=window)

    return rewards_history


# ---------------------------------------------------------------------------
# Цикл оценки
# ---------------------------------------------------------------------------

def evaluate(env: PacmanEnv, agent: SarsaAgent, output_dir: Path,
             episodes: int = 200, base_seed: int = 42) -> dict:
    """
    Запускает episodes эпизодов с seed = base_seed + i (i = 0..episodes-1).
    Считает:
      win_rate   — доля побед (побед / episodes * 100%)
      avg_reward — средняя суммарная награда за эпизод
      avg_steps  — средняя длина эпизода в шагах
    Сохраняет результат в eval_metrics.json.
    """
    print(f"\nEvaluating over {episodes} episodes "
          f"(seed {base_seed} .. {base_seed + episodes - 1})...")

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    wins:    list[bool]  = []
    rewards: list[float] = []
    lengths: list[int]   = []

    for i in range(episodes):
        episode_seed = base_seed + i
        obs, info    = env.reset(seed=episode_seed)
        state        = env.runtime_state
        total_reward = 0.0
        step_count   = 0
        done         = False

        while not done:
            legal  = info.get("legal_action_ids", [])
            action = agent.choose_action(state, legal)
            _, reward, terminated, truncated, info = env.step(action)
            done          = terminated or truncated
            state         = env.runtime_state
            total_reward += reward
            step_count   += 1

        wins.append(bool(info.get("is_win", False)))
        rewards.append(total_reward)
        lengths.append(step_count)

    agent.epsilon = saved_epsilon

    metrics = {
        "eval_episodes":     episodes,
        "base_seed":         base_seed,
        "seed_range":        f"{base_seed}..{base_seed + episodes - 1}",
        "wins":              int(sum(wins)),
        "win_rate_pct":      round(float(np.mean(wins)) * 100, 2),
        "avg_reward":        round(float(np.mean(rewards)), 2),
        "std_reward":        round(float(np.std(rewards)), 2),
        "avg_steps":         round(float(np.mean(lengths)), 2),
        "std_steps":         round(float(np.std(lengths)), 2),
        "min_reward":        round(float(np.min(rewards)), 2),
        "max_reward":        round(float(np.max(rewards)), 2),
    }

    print(f"\n{'='*50}")
    print(f"Eval results ({episodes} episodes, base_seed={base_seed}):")
    print(f"  Win rate:   {metrics['win_rate_pct']:.1f}%  ({metrics['wins']}/{episodes})")
    print(f"  Avg reward: {metrics['avg_reward']:.1f}  ± {metrics['std_reward']:.1f}")
    print(f"  Avg steps:  {metrics['avg_steps']:.1f}  ± {metrics['std_steps']:.1f}")
    print(f"{'='*50}\n")

    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Eval metrics saved to {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# GIF запись
# ---------------------------------------------------------------------------

def capture_frame():
    if not _PIL_AVAILABLE:
        return None
    canvas = graphicsUtils._canvas
    root   = graphicsUtils._root_window
    if canvas is None or root is None:
        return None
    root.update_idletasks()
    root.update()
    try:
        x0, y0 = int(canvas.winfo_rootx()), int(canvas.winfo_rooty())
        w, h   = int(canvas.winfo_width()), int(canvas.winfo_height())
        if w <= 1 or h <= 1:
            return None
        return ImageGrab.grab(bbox=(x0, y0, x0 + w, y0 + h))
    except Exception:
        return None


def save_gif(frames: list, output_path: Path, frame_time: float = 0.1) -> None:
    if not frames or not _PIL_AVAILABLE:
        return
    duration_ms = max(20, int(frame_time * 1000))
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   optimize=False, duration=duration_ms, loop=0)
    print(f"GIF saved to {output_path}")


def record_single_episode(env_config, agent: SarsaAgent,
                          seed: int, output_path: Path) -> None:
    if not _PIL_AVAILABLE:
        print("PIL not installed — cannot record GIF.")
        return

    from dataclasses import replace
    render_cfg = replace(env_config, render_mode="human")
    render_env = PacmanEnv(render_cfg, render_mode="human")

    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    gif_frames: list = []
    obs, info = render_env.reset(seed=seed)
    state     = render_env.runtime_state
    done      = False
    action    = agent.choose_action(state, info.get("legal_action_ids", []))

    frame = capture_frame()
    if frame:
        gif_frames.append(frame)

    while not done:
        _, _, terminated, truncated, info = render_env.step(action)
        state = render_env.runtime_state
        done  = terminated or truncated
        if not done:
            action = agent.choose_action(state, info.get("legal_action_ids", []))
        render_env.render()
        frame = capture_frame()
        if frame:
            gif_frames.append(frame)

    render_env.close()
    agent.epsilon = saved_epsilon

    output_path.parent.mkdir(exist_ok=True)
    save_gif(gif_frames, output_path, render_cfg.frame_time)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate SARSA(λ) for Pacman")
    parser.add_argument("--config",        type=str,   default="configs/default.yaml")
    parser.add_argument("--episodes",      type=int,   default=5000)
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--eval",          action="store_true")
    parser.add_argument("--eval-episodes", type=int,   default=200)
    parser.add_argument("--eval-seed",     type=int,   default=42)
    parser.add_argument("--output-dir",    type=str,   default="results/sarsa")
    parser.add_argument("--model",         type=str,   default=None)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--record",        action="store_true")
    parser.add_argument("--record-path",   type=str,   default=None)
    parser.add_argument("--lam",           type=float, default=0.8)
    parser.add_argument("--alpha",         type=float, default=0.01)
    parser.add_argument("--gamma",         type=float, default=0.9)
    parser.add_argument("--epsilon-end",   type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9998)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path  = Path(args.model) if args.model else output_dir / "sarsa_approx.pkl"
    record_path = Path(args.record_path) if args.record_path else output_dir / "replay.gif"

    config_dict = load_yaml(args.config)
    env_config  = build_env_config(config_dict.get("env", config_dict))
    base_seed   = env_config.seed

    print(f"Reward config: lose={env_config.reward.lose}, food={env_config.reward.food}")

    env   = PacmanEnv(env_config)
    agent = SarsaAgent(
        alpha   = args.alpha,
        gamma   = args.gamma,
        epsilon = 1.0,
        lam     = args.lam,
    )

    if args.train:
        train(env, agent, args.episodes,
              base_seed     = base_seed,
              output_dir    = output_dir,
              epsilon_end   = args.epsilon_end,
              epsilon_decay = args.epsilon_decay)
        agent.save(model_path)

    if args.eval:
        if not model_path.exists():
            print(f"Model file {model_path} not found!")
            return
        agent.load(model_path)
        evaluate(env, agent,
                 output_dir = output_dir,
                 episodes   = args.eval_episodes,
                 base_seed  = args.eval_seed)

    if args.record:
        if not model_path.exists():
            print(f"Model file {model_path} not found!")
            return
        agent.load(model_path)
        record_single_episode(
            env_config  = env_config,
            agent       = agent,
            seed        = args.eval_seed,
            output_path = record_path,
        )

    env.close()


if __name__ == "__main__":
    main()