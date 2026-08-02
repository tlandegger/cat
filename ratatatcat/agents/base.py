"""Agent protocol and trivial baselines."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional

from ..state import GameState, Move


class Agent(ABC):
    """A policy.

    Implementations must decide using only what ``player`` legitimately knows:
    ``state.knowledge[player]``, ``state.discard``, ``state.provenance`` and the
    public counters. Reading ``state.hands`` directly is cheating and will make
    benchmark numbers meaningless.
    """

    name: str = "agent"

    @abstractmethod
    def choose(self, state: GameState, player: int) -> Move:
        ...

    def reset(self) -> None:
        """Called at the start of each round. Override if the agent has state."""


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, state: GameState, player: int) -> Move:
        return self.rng.choice(state.legal_moves())


class NeverKnockRandomAgent(RandomAgent):
    """Random play that never calls, useful as a control in benchmarks."""

    name = "random-nocall"

    def choose(self, state: GameState, player: int) -> Move:
        from ..state import MoveKind

        moves: List[Move] = [m for m in state.legal_moves() if m.kind is not MoveKind.KNOCK]
        return self.rng.choice(moves or state.legal_moves())
