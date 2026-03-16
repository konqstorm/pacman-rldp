"""Policy wrapper that acts on aggregated observation keys."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..algorithms.policy_iteration.obs_encoding import encode_observation


class ObsPolicy:
    """Policy that selects actions from a precomputed table over observation keys."""

    def __init__(
        self,
        policy_table: dict[Any, int],
        *,
        seed: int | None = None,
        drop_keys: list[str] | None = None,
        float_round: int = 3,
        encoder: Callable[..., Any] | None = None,
    ) -> None:
        self.policy_table = policy_table
        self.drop_keys = drop_keys or []
        self.float_round = float_round
        self.encoder = encoder or encode_observation
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: dict[str, Any], info: dict[str, Any]) -> int:
        """Select action from policy table, fallback to random legal action."""
        key = self.encoder(observation, drop_keys=self.drop_keys, float_round=self.float_round)
        action = self.policy_table.get(key)
        legal_actions = info.get("legal_action_ids") if isinstance(info, dict) else None
        if action is None or (legal_actions and action not in legal_actions):
            if legal_actions:
                return int(self._rng.choice(legal_actions))
            return int(action) if action is not None else 4
        return int(action)
