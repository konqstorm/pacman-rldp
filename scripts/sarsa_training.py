import random
import pickle
import argparse
import time
from pathlib import Path
import numpy as np
import sys
from PIL import Image, ImageGrab

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


import random
import pickle
import argparse
import time
import sys
from pathlib import Path

import numpy as np

def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

_bootstrap_src_path()

from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.utils import load_yaml
from pacman_rldp.third_party.bk.util import manhattanDistance
from pacman_rldp.third_party.bk.game import Actions
from pacman_rldp.third_party.bk import graphicsUtils

try:
    from PIL import Image, ImageGrab
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Извлечение ключа состояния из runtime GameState
# ---------------------------------------------------------------------------

def _dist_bucket(dist: float) -> int:
    """Манхэттенское расстояние → бакет: 0=вплотную, 1=близко, 2=далеко."""
    if dist <= 2:
        return 0
    if dist <= 6:
        return 1
    return 2


def _direction_sign(src: tuple, dst: tuple) -> tuple[int, int]:
    """Знак разности координат: каждая компонента из {-1, 0, 1}."""
    return (
        int(np.sign(dst[0] - src[0])),
        int(np.sign(dst[1] - src[1])),
    )


def _bfs_nearest(start: tuple, targets: list, walls) -> tuple[float, tuple | None]:
    """BFS по сетке — возвращает (расстояние, позиция) ближайшей цели."""
    if not targets:
        return float("inf"), None
    target_set = set(map(tuple, targets))
    fringe = [(start[0], start[1], 0)]
    visited: set = set()
    while fringe:
        x, y, d = fringe.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        if (x, y) in target_set:
            return d, (x, y)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                fringe.append((nx, ny, d + 1))
    return float("inf"), None

def _direction_4(src: tuple, dst: tuple | None) -> int:
    """Преобладающее направление: 0=нет, 1=север, 2=юг, 3=восток, 4=запад"""
    if dst is None:
        return 0
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) >= abs(dy):
        return 3 if dx > 0 else 4
#     return 1 if dy > 0 else 2

def _bfs_nearest_with_first_step(
    start: tuple, targets: list, walls
) -> tuple[float, tuple | None, tuple | None]:
    """BFS — возвращает (расстояние, цель, первый шаг)."""
    if not targets:
        return float("inf"), None, None
    target_set = set(map(tuple, targets))
    # parent хранит откуда пришли
    parent: dict[tuple, tuple | None] = {start: None}
    fringe = [(start[0], start[1], 0)]
    visited: set = set()
    found = None
    while fringe:
        x, y, d = fringe.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        if (x, y) in target_set:
            found = (x, y, d)
            break
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < walls.width and 0 <= ny < walls.height
                    and not walls[nx][ny] and (nx, ny) not in parent):
                parent[(nx, ny)] = (x, y)
                fringe.append((nx, ny, d + 1))
    if found is None:
        return float("inf"), None, None
    # Восстановить первый шаг
    node = (found[0], found[1])
    while parent[node] != start and parent[node] is not None:
        node = parent[node]
    return found[2], (found[0], found[1]), node

def get_state_key(state) -> tuple:
    pac_pos  = state.getPacmanPosition()
    pac_int  = (int(pac_pos[0]), int(pac_pos[1]))
    food     = state.getFood()
    walls    = state.getWalls()
    ghosts   = state.getGhostStates()
    capsules = state.getCapsules()
    norm     = float(walls.width + walls.height)

    # --- Еда: направление первого BFS-шага + бакет расстояния ---
    food_list = food.asList()
    food_count_b = 0 if len(food_list) <= 3 else (1 if len(food_list) <= 10 else 2)

    if food_list:
        food_dist, _, first_step = _bfs_nearest_with_first_step(pac_int, food_list, walls)
        food_dir    = _direction_4(pac_int, first_step)
        raw_f       = food_dist / norm if food_dist < float("inf") else 1.0
        food_bucket = 0 if raw_f < 0.05 else (1 if raw_f < 0.15 else (2 if raw_f < 0.30 else 3))
    else:
        food_dir, food_bucket = 0, 4

    # --- Еда рядом: плотность в 3×3 ---
    food_nearby = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x, y = pac_int[0] + dx, pac_int[1] + dy
            if 0 <= x < walls.width and 0 <= y < walls.height and food[x][y]:
                food_nearby += 1
    food_nearby_b = 0 if food_nearby == 0 else (1 if food_nearby <= 3 else 2)

    # --- Призрак: направление + бакет расстояния + опасность + испуг ---
    active_ghosts = [gs for gs in ghosts if gs.getPosition() is not None]
    if active_ghosts:
        nearest_gs   = min(active_ghosts,
                           key=lambda gs: manhattanDistance(pac_pos, gs.getPosition()))
        g_pos        = nearest_gs.getPosition()
        g_dist       = manhattanDistance(pac_pos, g_pos)
        ghost_dir    = _direction_4(pac_int, (int(g_pos[0]), int(g_pos[1])))
        raw_g        = g_dist / norm
        ghost_bucket = 0 if raw_g < 0.05 else (1 if raw_g < 0.15 else (2 if raw_g < 0.30 else 3))
        danger       = int(g_dist <= 1)
        scared       = int(nearest_gs.scaredTimer > 0)
    else:
        ghost_dir, ghost_bucket, danger, scared = 0, 3, 0, 0

    # --- Капсула: только бакет расстояния ---
    if capsules:
        cap_dist  = min(manhattanDistance(pac_pos, c) for c in capsules)
        raw_c     = cap_dist / norm
        cap_dist_b = 0 if raw_c < 0.05 else (1 if raw_c < 0.20 else 2)
    else:
        cap_dist_b = 3

    return (
        food_dir,       # 0–4: куда идти за едой
        food_bucket,    # 0–4: как далеко еда
        food_nearby_b,  # 0–2: плотность еды рядом
        food_count_b,   # 0–2: сколько еды осталось
        ghost_dir,      # 0–4: откуда идёт призрак
        ghost_bucket,   # 0–3: как далеко призрак
        danger,         # 0/1: призрак вплотную
        scared,         # 0/1: призрак испуган
        cap_dist_b,     # 0–3: расстояние до капсулы
    )

# ---------------------------------------------------------------------------
# Табличный SARSA агент
# ---------------------------------------------------------------------------

class SarsaAgent:
    """Табличный SARSA агент с runtime GameState."""

    ACTION_MAP = ["North", "South", "East", "West", "Stop"]

    def __init__(self, action_space_size: int = 5, alpha: float = 0.3,
                 gamma: float = 0.95, epsilon: float = 1.0):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size
        self.q_table: dict[tuple, np.ndarray] = {}

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        return self.q_table[state_key]

    def choose_action(self, state, legal_actions: list[int]) -> int:
        # Убираем STOP из легальных действий чтобы агент не стоял на месте
        legal = [a for a in legal_actions if a != 4]
        if not legal:
            legal = legal_actions

        if random.random() < self.epsilon:
            return random.choice(legal)

        state_key = get_state_key(state)
        q_values  = self._get_q_values(state_key)

        masked = np.full(self.action_space_size, -np.inf)
        for a in legal:
            masked[a] = q_values[a]

        max_q        = np.max(masked)
        best_actions = [a for a in legal if masked[a] == max_q]
        return random.choice(best_actions)

    def learn(self, state, action: int, reward: float,
              next_state, next_action: int | None, done: bool) -> None:
        """Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]"""
        s_key  = get_state_key(state)
        current_q = float(self._get_q_values(s_key)[action])

        if done or next_action is None:
            next_q = 0.0
        else:
            ns_key = get_state_key(next_state)
            next_q = float(self._get_q_values(ns_key)[next_action])

        td_error = reward + self.gamma * next_q - current_q
        self._get_q_values(s_key)[action] += self.alpha * td_error

    def save(self, path: str | Path) -> None:
        serializable = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(path, "wb") as f:
            pickle.dump({"q_table": serializable, "alpha": self.alpha,
                         "gamma": self.gamma}, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        import ast
        self.alpha   = data.get("alpha", self.alpha)
        self.gamma   = data.get("gamma", self.gamma)
        self.q_table = {
            ast.literal_eval(k): np.array(v)
            for k, v in data["q_table"].items()
        }
        print(f"Policy loaded. States: {len(self.q_table)}")


# ---------------------------------------------------------------------------
# Цикл обучения
# ---------------------------------------------------------------------------

def train(env: PacmanEnv, agent: SarsaAgent, episodes: int,
          epsilon_end: float = 0.05, epsilon_decay: float = 0.9995) -> list[float]:
    print(f"Starting training for {episodes} episodes...")
    rewards_history: list[float] = []
    wins_history: list[bool] = []          # добавить

    for i in range(episodes):
        obs, info = env.reset()
        state        = env.runtime_state
        total_reward = 0.0
        done         = False

        legal_actions = info.get("legal_action_ids", [])
        action        = agent.choose_action(state, legal_actions)

        while not done:
            _, reward, terminated, truncated, info = env.step(action)
            done         = terminated or truncated
            next_state   = env.runtime_state
            total_reward += reward

            if done:
                agent.learn(state, action, reward, next_state, None, done)
            else:
                next_legal  = info.get("legal_action_ids", [])
                next_action = agent.choose_action(next_state, next_legal)
                agent.learn(state, action, reward, next_state, next_action, done)
                action = next_action

            state = next_state

        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        wins_history.append(bool(info.get("is_win", False)))  # добавить

        if (i + 1) % 100 == 0:
            avg      = np.mean(rewards_history[-100:])
            win_rate = np.mean(wins_history[-100:]) * 100   # добавить
            print(f"Episode {i+1}/{episodes} | "
                  f"Avg(100): {avg:.1f} | "
                  f"WinRate(100): {win_rate:.1f}% | "      # добавить
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"States: {len(agent.q_table)}")

    return rewards_history

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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular SARSA Agent for Pacman")
    parser.add_argument("--config",      type=str,  default="configs/default.yaml")
    parser.add_argument("--episodes",    type=int,  default=3000)
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--model",       type=str,  default="results/sarsa_table.pkl")
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--seed",        type=int,  default=None)
    parser.add_argument("--record",      action="store_true")
    parser.add_argument("--record-path", type=str,  default="results/replay.gif")
    args = parser.parse_args()

    # config_dict = load_yaml(args.config)
    # env_config  = build_env_config(config_dict)
    config_dict = load_yaml(args.config)
    env_config  = build_env_config(config_dict.get("env", config_dict))
    print(f"Reward config: lose={env_config.reward.lose}, food={env_config.reward.food}")

    if args.seed is not None:
        env_config.seed = args.seed

    env   = PacmanEnv(env_config)
    agent = SarsaAgent()

    if args.train:
        train(env, agent, args.episodes)
        Path("results").mkdir(exist_ok=True)
        agent.save(args.model)
        print(f"Model saved to {args.model}")

    if args.eval:
        if not Path(args.model).exists():
            print(f"Model file {args.model} not found!")
            return
        agent.load(args.model)
        agent.epsilon = 0.0

        from dataclasses import replace
        render_cfg = replace(env_config, render_mode="human")
        render_env = PacmanEnv(render_cfg, render_mode="human")

        gif_frames: list = []
        obs, info = render_env.reset(seed=env_config.seed)
        state     = render_env.runtime_state
        done      = False
        action    = agent.choose_action(state, info.get("legal_action_ids", []))

        if args.record:
            frame = capture_frame()
            if frame:
                gif_frames.append(frame)

        while not done:
            _, _, terminated, truncated, info = render_env.step(action)
            state = render_env.runtime_state
            done  = terminated or truncated

            if not done:
                action = agent.choose_action(state, info.get("legal_action_ids", []))

            if args.render or args.record:
                render_env.render()
                if args.record:
                    frame = capture_frame()
                    if frame:
                        gif_frames.append(frame)
                else:
                    time.sleep(0.05)

        render_env.close()
        print("Evaluation finished.")

        if args.record and gif_frames:
            output_path = Path(args.record_path)
            output_path.parent.mkdir(exist_ok=True)
            save_gif(gif_frames, output_path, env_config.frame_time)

    env.close()


if __name__ == "__main__":
    main()