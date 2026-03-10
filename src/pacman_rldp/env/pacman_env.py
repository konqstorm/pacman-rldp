"""Gymnasium-style Pacman environment built on the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..third_party.bk import game as runtime_game
from ..third_party.bk import graphicsDisplay
from ..third_party.bk import layout as runtime_layout
from ..third_party.bk import pacman as runtime_pacman


@dataclass(frozen=True)
class RewardConfig:
    """Event-to-reward mapping for environment transitions."""

    time_penalty: float = -1.0
    food: float = 10.0
    capsule: float = 0.0
    eat_ghost: float = 200.0
    win: float = 500.0
    lose: float = -500.0
    invalid_action: float = -5.0


@dataclass
class PacmanEnvConfig:
    """Configuration values that define environment dynamics and rendering."""

    layout_name: str = "smallClassic"
    num_ghosts: int = 2
    max_steps: int = 500
    seed: int = 42
    ghost_policy: str = "random"
    invalid_action_mode: str = "raise"
    render_mode: str | None = None
    zoom: float = 1.0
    frame_time: float = 0.1
    reward: RewardConfig = field(default_factory=RewardConfig)


class PacmanEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium wrapper around the refactored Pacman runtime."""

    metadata = {"render_modes": ["human", "ansi", None], "render_fps": 10}

    _ACTIONS: tuple[str, ...] = (
        runtime_game.Directions.NORTH,
        runtime_game.Directions.SOUTH,
        runtime_game.Directions.EAST,
        runtime_game.Directions.WEST,
        runtime_game.Directions.STOP,
    )

    def __init__(self, config: PacmanEnvConfig | None = None, render_mode: str | None = None) -> None:
        """Initialize the environment with validated config and spaces."""
        super().__init__()
        self.config = config or PacmanEnvConfig()
        self.render_mode = render_mode if render_mode is not None else self.config.render_mode

        self._layout = runtime_layout.getLayout(self.config.layout_name)
        if self._layout is None:
            raise ValueError(f"Unknown layout '{self.config.layout_name}'.")

        self._ghost_count = min(self.config.num_ghosts, self._layout.getNumGhosts())
        self.action_space = spaces.Discrete(len(self._ACTIONS))
        self.observation_space = self._build_observation_space()

        self._seed_value = self.config.seed
        self._rng = np.random.default_rng(self._seed_value)
        self._state: runtime_pacman.GameState | None = None
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._display: Any | None = None
        self._display_initialized = False
        self.seed(self._seed_value)

    def seed(self, seed: int | None = None) -> None:
        """Set deterministic random generators for environment and ghost sampling."""
        if seed is None:
            seed = self._seed_value
        self._seed_value = int(seed)
        self._rng = np.random.default_rng(self._seed_value)
        random.seed(self._seed_value)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the episode and return initial observation and metadata."""
        del options
        if seed is not None:
            self.seed(seed)

        state = runtime_pacman.GameState()
        state.initialize(self._layout, numGhostAgents=self._ghost_count)

        self._state = state
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        if self.render_mode == "human":
            self._reset_human_display()
            self.render()

        observation = self._build_observation(self._state)
        info = self._build_info(self._state)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance the environment by one Pacman action and one ghost-response phase."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is finished. Call reset() before stepping again.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action id {action} is out of bounds.")

        requested_direction = self._ACTIONS[action]
        legal_actions = self._state.getLegalActions(0)
        invalid_action = requested_direction not in legal_actions

        selected_direction = requested_direction
        if invalid_action:
            if self.config.invalid_action_mode == "raise":
                raise ValueError(
                    f"Illegal action '{requested_direction}' for current state. Legal: {legal_actions}"
                )
            stop_direction = runtime_game.Directions.STOP
            selected_direction = stop_direction if stop_direction in legal_actions else legal_actions[0]

        state_before = self.clone_state(self._state)
        transition_states: list[runtime_pacman.GameState] = []
        next_state = self._state.generateSuccessor(0, selected_direction)
        transition_states.append(next_state)

        if not (next_state.isWin() or next_state.isLose()):
            for ghost_index in range(1, next_state.getNumAgents()):
                if next_state.isWin() or next_state.isLose():
                    break
                ghost_legal_actions = next_state.getLegalActions(ghost_index)
                if not ghost_legal_actions:
                    continue
                ghost_action = self._sample_ghost_action(ghost_legal_actions)
                next_state = next_state.generateSuccessor(ghost_index, ghost_action)
                transition_states.append(next_state)

        reward, events = self.compute_reward_from_transition(
            state_before=state_before,
            state_after=next_state,
            invalid_action=invalid_action,
        )

        self._state = next_state
        self._step_count += 1
        self._terminated = bool(next_state.isWin() or next_state.isLose())
        self._truncated = bool((not self._terminated) and (self._step_count >= self.config.max_steps))

        if self.render_mode == "human":
            for transition_state in transition_states:
                self._render_runtime_state(transition_state)

        observation = self._build_observation(self._state)
        info = self._build_info(self._state)
        info["events"] = events
        info["selected_direction"] = selected_direction
        info["requested_direction"] = requested_direction
        info["invalid_action"] = invalid_action

        return observation, reward, self._terminated, self._truncated, info

    def render(self) -> str | None:
        """Render state using human (Tk) or ANSI text mode."""
        if self._state is None:
            return None
        if self.render_mode == "ansi":
            return str(self._state)
        if self.render_mode == "human":
            self._render_runtime_state(self._state)
            return None
        return None

    def close(self) -> None:
        """Release renderer resources if the display was initialized."""
        if self._display is not None:
            self._display.finish()
        self._display = None
        self._display_initialized = False

    @property
    def runtime_state(self) -> runtime_pacman.GameState:
        """Return a deep-copied runtime state for external planners."""
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self.clone_state(self._state)

    def legal_action_ids(self, state: runtime_pacman.GameState | None = None) -> list[int]:
        """Return legal action ids for Pacman in the provided or current state."""
        current = state if state is not None else self._state
        if current is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        legal_directions = current.getLegalActions(0)
        return [idx for idx, direction in enumerate(self._ACTIONS) if direction in legal_directions]

    def action_id_to_direction(self, action_id: int) -> str:
        """Convert integer action id into direction string."""
        if not self.action_space.contains(action_id):
            raise ValueError(f"Action id {action_id} is out of bounds.")
        return self._ACTIONS[action_id]

    def compute_reward_from_transition(
        self,
        state_before: runtime_pacman.GameState,
        state_after: runtime_pacman.GameState,
        invalid_action: bool,
    ) -> tuple[float, dict[str, int]]:
        """Compute configured reward using event markers from the runtime state."""
        del state_before
        eaten_flags = state_after.data._eaten if state_after.data._eaten is not None else []
        ghost_eaten = int(sum(1 for eaten in eaten_flags[1:] if eaten))
        events = {
            "food_eaten": int(state_after.data._foodEaten is not None),
            "capsule_eaten": int(state_after.data._capsuleEaten is not None),
            "ghost_eaten": ghost_eaten,
            "win": int(state_after.isWin()),
            "lose": int(state_after.isLose()),
            "invalid_action": int(invalid_action),
        }

        reward_cfg = self.config.reward
        reward = float(reward_cfg.time_penalty)
        reward += float(events["food_eaten"]) * reward_cfg.food
        reward += float(events["capsule_eaten"]) * reward_cfg.capsule
        reward += float(events["ghost_eaten"]) * reward_cfg.eat_ghost
        reward += float(events["win"]) * reward_cfg.win
        reward += float(events["lose"]) * reward_cfg.lose
        reward += float(events["invalid_action"]) * reward_cfg.invalid_action
        return reward, events

    @staticmethod
    def clone_state(state: runtime_pacman.GameState) -> runtime_pacman.GameState:
        """Create a deep-copied runtime game state."""
        return runtime_pacman.GameState(state)

    def _sample_ghost_action(self, legal_actions: list[str]) -> str:
        """Sample a ghost action using configured stochastic policy."""
        if self.config.ghost_policy != "random":
            raise ValueError(f"Unsupported ghost policy '{self.config.ghost_policy}'.")
        sampled_index = int(self._rng.integers(low=0, high=len(legal_actions)))
        return legal_actions[sampled_index]

    def _build_observation_space(self) -> spaces.Dict:
        """Build observation space that matches structured dictionary output."""
        width = int(self._layout.width)
        height = int(self._layout.height)
        max_coord = float(max(width, height))
        return spaces.Dict(
            {
                "pacman_position": spaces.Box(
                    low=np.array([-1.0, -1.0], dtype=np.float32),
                    high=np.array([max_coord, max_coord], dtype=np.float32),
                    dtype=np.float32,
                ),
                "ghost_positions": spaces.Box(
                    low=-1.0,
                    high=max_coord,
                    shape=(self._ghost_count, 2),
                    dtype=np.float32,
                ),
                "ghost_timers": spaces.Box(
                    low=0,
                    high=999,
                    shape=(self._ghost_count,),
                    dtype=np.int32,
                ),
                "ghost_present": spaces.MultiBinary(self._ghost_count),
                "walls": spaces.Box(
                    low=0,
                    high=1,
                    shape=(width, height),
                    dtype=np.int8,
                ),
                "food": spaces.Box(
                    low=0,
                    high=1,
                    shape=(width, height),
                    dtype=np.int8,
                ),
                "capsules": spaces.Box(
                    low=0,
                    high=1,
                    shape=(width, height),
                    dtype=np.int8,
                ),
                "score": spaces.Box(
                    low=-1_000_000.0,
                    high=1_000_000.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "step_count": spaces.Box(
                    low=0,
                    high=self.config.max_steps,
                    shape=(1,),
                    dtype=np.int32,
                ),
            }
        )

    def _build_observation(self, state: runtime_pacman.GameState) -> dict[str, np.ndarray]:
        """Create structured observation dictionary from runtime state."""
        width = int(self._layout.width)
        height = int(self._layout.height)

        pac_pos = state.getPacmanPosition()
        pacman_position = np.array([float(pac_pos[0]), float(pac_pos[1])], dtype=np.float32)

        ghost_positions = np.full((self._ghost_count, 2), fill_value=-1.0, dtype=np.float32)
        ghost_timers = np.zeros((self._ghost_count,), dtype=np.int32)
        ghost_present = np.zeros((self._ghost_count,), dtype=np.int8)

        for ghost_idx, ghost_state in enumerate(state.getGhostStates()[: self._ghost_count]):
            ghost_position = ghost_state.getPosition()
            if ghost_position is None:
                continue
            ghost_positions[ghost_idx] = np.array(
                [float(ghost_position[0]), float(ghost_position[1])],
                dtype=np.float32,
            )
            ghost_timers[ghost_idx] = int(ghost_state.scaredTimer)
            ghost_present[ghost_idx] = 1

        walls = self._grid_to_binary_array(state.getWalls())
        food = self._grid_to_binary_array(state.getFood())
        capsules = np.zeros((width, height), dtype=np.int8)
        for capsule_x, capsule_y in state.getCapsules():
            capsules[int(capsule_x), int(capsule_y)] = 1

        observation = {
            "pacman_position": pacman_position,
            "ghost_positions": ghost_positions,
            "ghost_timers": ghost_timers,
            "ghost_present": ghost_present,
            "walls": walls,
            "food": food,
            "capsules": capsules,
            "score": np.array([float(state.getScore())], dtype=np.float32),
            "step_count": np.array([self._step_count], dtype=np.int32),
        }
        return observation

    def _build_info(self, state: runtime_pacman.GameState) -> dict[str, Any]:
        """Build step metadata that is useful for algorithms and debugging."""
        legal_action_ids = self.legal_action_ids(state)
        return {
            "legal_action_ids": legal_action_ids,
            "legal_directions": [self._ACTIONS[idx] for idx in legal_action_ids],
            "score": float(state.getScore()),
            "is_win": bool(state.isWin()),
            "is_lose": bool(state.isLose()),
            "step_count": int(self._step_count),
            "seed": int(self._seed_value),
        }

    @staticmethod
    def _grid_to_binary_array(grid: runtime_game.Grid) -> np.ndarray:
        """Convert boolean grid object into int8 array."""
        return np.asarray(grid.data, dtype=np.int8)

    def _ensure_human_display(self) -> None:
        """Initialize graphical renderer object on demand."""
        if self._display is None:
            self._display = graphicsDisplay.PacmanGraphics(
                self.config.zoom,
                frameTime=self.config.frame_time,
            )
            self._display_initialized = False

    def _reset_human_display(self) -> None:
        """Reset graphical renderer state at episode boundaries."""
        if self._display is not None:
            self._display.finish()
        self._display = None
        self._display_initialized = False

    def _render_runtime_state(self, state: runtime_pacman.GameState) -> None:
        """Render one concrete runtime state in the human display."""
        self._ensure_human_display()
        if not self._display_initialized:
            self._display.initialize(state.data)
            self._display_initialized = True
        else:
            self._display.update(state.data)


def build_env_config(config_dict: dict[str, Any]) -> PacmanEnvConfig:
    """Build typed environment config from a raw dictionary."""
    reward_dict = config_dict.get("reward", {})
    reward = RewardConfig(
        time_penalty=float(reward_dict.get("time_penalty", -1.0)),
        food=float(reward_dict.get("food", 10.0)),
        capsule=float(reward_dict.get("capsule", 0.0)),
        eat_ghost=float(reward_dict.get("eat_ghost", 200.0)),
        win=float(reward_dict.get("win", 500.0)),
        lose=float(reward_dict.get("lose", -500.0)),
        invalid_action=float(reward_dict.get("invalid_action", -5.0)),
    )
    return PacmanEnvConfig(
        layout_name=str(config_dict.get("layout_name", "smallClassic")),
        num_ghosts=int(config_dict.get("num_ghosts", 2)),
        max_steps=int(config_dict.get("max_steps", 500)),
        seed=int(config_dict.get("seed", 42)),
        ghost_policy=str(config_dict.get("ghost_policy", "random")),
        invalid_action_mode=str(config_dict.get("invalid_action_mode", "raise")),
        render_mode=config_dict.get("render_mode"),
        zoom=float(config_dict.get("zoom", 1.0)),
        frame_time=float(config_dict.get("frame_time", 0.1)),
        reward=reward,
    )
