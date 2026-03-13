import random
import pickle
import argparse
import time
from pathlib import Path
import numpy as np
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.utils import load_yaml
from pacman_rldp.third_party.bk.util import Counter, manhattanDistance
from pacman_rldp.third_party.bk.game import Actions
from pacman_rldp.third_party.bk import graphicsUtils
from PIL import Image, ImageGrab


class SimpleFeatureExtractor:
    """Извлекает базовые признаки для Approximate Q-learning."""
    def get_features(self, state, action):
        features = Counter()
        features["bias"] = 1.0
        
        pacman_pos = state.getPacmanPosition()
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)
        
        # Если идем в стену - очень плохо
        if next_x < 0 or next_x >= walls.width or next_y < 0 or next_y >= walls.height or walls[next_x][next_y]:
            return features # Возвращаем только bias
            
        # Признак: есть ли еда в клетке, куда идем
        if food[next_x][next_y]:
            features["eats-food"] = 1.0
            
        # Признак: расстояние до ближайшей еды
        dist = self.closest_food((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)
            
        # Признак: близость призраков
        for ghost_pos in ghosts:
            if manhattanDistance((next_x, next_y), ghost_pos) <= 1:
                features["ghost-nearby"] = 1.0
                
        return features

    def closest_food(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded: continue
            expanded.add((pos_x, pos_y))
            if food[pos_x][pos_y]: return dist
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = pos_x + dx, pos_y + dy
                if nx >= 0 and nx < walls.width and ny >= 0 and ny < walls.height and not walls[nx][ny]:
                    fringe.append((nx, ny, dist + 1))
        return None


class QLearningAgent:
    """Approximate Q-learning Agent."""
    def __init__(self, action_space_size, alpha=0.01, gamma=0.9, epsilon=0.1):
        self.weights = Counter()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_extractor = SimpleFeatureExtractor()

    def get_q_value(self, state, action_id):
        action = ["North", "South", "East", "West", "Stop"][action_id]
        features = self.feature_extractor.get_features(state, action)
        return features * self.weights

    def choose_action(self, state, legal_actions):
        if not legal_actions:
            return 4
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        q_values = {a: self.get_q_value(state, a) for a in legal_actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        q_value = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            # V(s') = max_a Q(s', a) для легальных действий
            legal_next = next_state.getLegalPacmanActions()
            if not legal_next:
                next_max_q = 0
            else:
                action_map = ["North", "South", "East", "West", "Stop"]
                q_values = [self.get_q_value(next_state, action_map.index(a)) for a in legal_next]
                next_max_q = max(q_values)
            target = reward + self.gamma * next_max_q
            
        diff = target - q_value
        action_str = ["North", "South", "East", "West", "Stop"][action]
        features = self.feature_extractor.get_features(state, action_str)
        for feature, value in features.items():
            self.weights[feature] += self.alpha * diff * value

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.weights), f)

    def load(self, path):
        with open(path, "rb") as f:
            w_dict = pickle.load(f)
            self.weights = Counter(w_dict)


def capture_human_frame() -> Image.Image | None:
    """Capture one frame from the active Tk canvas."""
    canvas = graphicsUtils._canvas
    root_window = graphicsUtils._root_window
    if canvas is None or root_window is None:
        return None

    root_window.update_idletasks()
    root_window.update()
    try:
        x0 = int(canvas.winfo_rootx())
        y0 = int(canvas.winfo_rooty())
        width = int(canvas.winfo_width())
        height = int(canvas.winfo_height())
        if width <= 1 or height <= 1:
            return None
        return ImageGrab.grab(bbox=(x0, y0, x0 + width, y0 + height))
    except Exception:
        return None


def save_gif(frames: list[Image.Image], output_path: Path, frame_time: float) -> None:
    """Save accumulated frames into an animated GIF."""
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


def train(env, agent, episodes):
    print(f"Starting training for {episodes} episodes...")
    rewards_history = []
    
    for i in range(episodes):
        obs, info = env.reset()
        state = env.runtime_state
        total_reward = 0
        done = False
        
        while not done:
            legal_actions = info.get("legal_action_ids", [])
            action = agent.choose_action(state, legal_actions)
            _, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.runtime_state
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        if (i + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {i+1}/{episodes}, Avg Reward: {avg_reward:.2f}, Weights: {dict(agent.weights)}")

def main():
    parser = argparse.ArgumentParser(description="Q-Learning Agent for Pacman")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--train", action="store_true", help="Run in training mode")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    parser.add_argument("--model", type=str, default="q_weights.pkl", help="Path to save/load model")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment")
    parser.add_argument("--record", action="store_true", help="Record frames for GIF")
    parser.add_argument("--record-dir", type=str, default="frames", help="Directory to save frames")
    args = parser.parse_args()

    config_dict = load_yaml(args.config)
    env_config = build_env_config(config_dict)
    
    # Override seed if provided via CLI
    if args.seed is not None:
        env_config.seed = args.seed
        
    # Force human render mode if requested
    if args.render or args.record:
        env_config.render_mode = "human"
        
    env = PacmanEnv(env_config)

    agent = QLearningAgent(action_space_size=5)

    if args.train:
        train(env, agent, args.episodes)
        Path("results").mkdir(exist_ok=True) # Ensure results directory exists
        agent.save(args.model)
        print(f"Model saved to {args.model}")

    if args.eval:
        if not Path(args.model).exists():
            print(f"Model file {args.model} not found!")
            return
        agent.load(args.model)
        agent.epsilon = 0.0
        
        if args.record:
            gif_frames = []
            print(f"Recording enabled. Frames will be captured during evaluation.")

        print(f"Evaluating model {args.model} using seed {env_config.seed}...")
        obs, info = env.reset(seed=env_config.seed)
        state = env.runtime_state
        done = False
        
        if args.record:
            frame = capture_human_frame()
            if frame: gif_frames.append(frame)

        while not done:
            action = agent.choose_action(state, info.get("legal_action_ids", []))
            _, _, terminated, truncated, info = env.step(action)
            state = env.runtime_state
            done = terminated or truncated
            if args.render or args.record:
                env.render()
                if args.record:
                    frame = capture_human_frame()
                    if frame: gif_frames.append(frame)
                else:
                    time.sleep(0.05)
        
        print("Evaluation finished.")
        if args.record:
            output_path = Path("results") / (args.record_dir + ".gif")
            output_path.parent.mkdir(exist_ok=True)
            save_gif(gif_frames, output_path, env_config.frame_time)
            print(f"GIF saved to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
