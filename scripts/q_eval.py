import argparse
from pathlib import Path
import pickle
import numpy as np
from pacman_rldp.env import PacmanEnv, build_env_config
from pacman_rldp.third_party.bk.util import Counter, manhattanDistance
from pacman_rldp.third_party.bk.game import Actions

class QLearningPolicy:
    def __init__(self, weights_path):
        with open(weights_path, "rb") as f:
            w_dict = pickle.load(f)
        self.weights = Counter(w_dict)
        self.action_map = ["North", "South", "East", "West", "Stop"]

    def get_features(self, state, action):
        features = Counter()
        features["bias"] = 1.0
        pacman_pos = state.getPacmanPosition()
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(pacman_pos[0] + dx), int(pacman_pos[1] + dy)
        if next_x < 0 or next_x >= walls.width or next_y < 0 or next_y >= walls.height or walls[next_x][next_y]:
            return features
        if food[next_x][next_y]:
            features["eats-food"] = 1.0
        dist = self.closest_food((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)
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

    def get_q_value(self, state, action_id):
        action = self.action_map[action_id]
        features = self.get_features(state, action)
        return features * self.weights

    def select_action(self, state, legal_actions):
        if not legal_actions:
            return 4
        q_values = {a: self.get_q_value(state, a) for a in legal_actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return np.random.choice(best_actions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="q_weights.pkl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from pacman_rldp.utils import load_yaml
    config_dict = load_yaml(args.config)
    env_config = build_env_config(config_dict)
    env = PacmanEnv(env_config)
    policy = QLearningPolicy(args.model)

    wins = 0
    for i in range(args.episodes):
        obs, info = env.reset(seed=args.seed + i)
        state = env.runtime_state
        done = False
        while not done:
            legal_actions = info.get("legal_action_ids", [])
            action = policy.select_action(state, legal_actions)
            _, _, terminated, truncated, info = env.step(action)
            state = env.runtime_state
            done = terminated or truncated
        if info.get("is_win", False):
            wins += 1
        print(f"Episode {i+1}: win={info.get('is_win', False)}, score={info.get('score', 0)}")
    print(f"Win rate: {wins / args.episodes:.3f}")

if __name__ == "__main__":
    main()