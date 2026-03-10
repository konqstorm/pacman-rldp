"""Manual gameplay helpers using Berkeley keyboard control and displays."""

from __future__ import annotations

import random

from ..third_party.bk import ghostAgents
from ..third_party.bk import graphicsDisplay
from ..third_party.bk import keyboardAgents
from ..third_party.bk import layout as runtime_layout
from ..third_party.bk import pacman as runtime_pacman
from ..third_party.bk import textDisplay


def run_keyboard_game(
    layout_name: str,
    num_ghosts: int,
    seed: int,
    render_mode: str,
    zoom: float,
    frame_time: float,
) -> float:
    """Run one interactive keyboard game and return the final score."""
    if render_mode != "human":
        raise ValueError("Keyboard-driven manual mode requires render_mode='human'.")

    random.seed(seed)
    chosen_layout = runtime_layout.getLayout(layout_name)
    if chosen_layout is None:
        raise ValueError(f"Unknown layout '{layout_name}'.")

    pacman_agent = keyboardAgents.KeyboardAgent(index=0)
    ghost_count = min(num_ghosts, chosen_layout.getNumGhosts())
    ghosts = [ghostAgents.RandomGhost(index + 1) for index in range(ghost_count)]

    display = graphicsDisplay.PacmanGraphics(zoom=zoom, frameTime=frame_time)
    rules = runtime_pacman.ClassicGameRules()
    game = rules.newGame(
        layout=chosen_layout,
        pacmanAgent=pacman_agent,
        ghostAgents=ghosts,
        display=display,
        quiet=False,
        catchExceptions=False,
    )
    game.run()
    return float(game.state.getScore())


def build_text_display(frame_time: float) -> textDisplay.PacmanGraphics:
    """Construct Berkeley text display with explicit frame timing."""
    return textDisplay.PacmanGraphics(speed=frame_time)
