"""A strong rule-based player.

This is both a competitive opponent and the rollout policy for the search
agents -- MCTS quality depends heavily on rollouts being better than random, so
this file matters more than its simplicity suggests.

The core quantity is the *slot estimate*: the known face value if the player
knows it, otherwise the belief-weighted expected value. Every decision reduces
to comparing a candidate card against the worst slot estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..belief import expected_slot_value, hand_estimates
from ..cards import Card
from ..state import GameState, Move, MoveKind, Phase, other
from .base import Agent


@dataclass
class HeuristicParams:
    #: Take a card from the discard only if it beats our worst slot by this much.
    take_discard_margin: float = 0.5
    #: Keep a drawn number card only if it beats our worst slot by this much.
    keep_margin: float = 0.5
    #: Call only when we expect to be ahead by at least this many points.
    knock_lead: float = 4.0
    #: Do not call while this much of our own hand is still unknown (points of
    #: uncertainty, summed over unknown slots).
    knock_max_unknown_slots: int = 1
    #: Swap only when the expected gain clears this bar.
    swap_margin: float = 1.5


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, params: Optional[HeuristicParams] = None) -> None:
        self.p = params or HeuristicParams()

    # ---------------------------------------------------------------- helpers
    def _estimates(self, state: GameState, player: int, owner: int) -> List[float]:
        """Per-slot expected point cost, from ``player``'s point of view."""
        return hand_estimates(state, player, owner)

    def _worst_slot(self, state: GameState, player: int) -> Tuple[int, float]:
        est = self._estimates(state, player, player)
        idx = max(range(len(est)), key=lambda s: est[s])
        return idx, est[idx]

    def _unknown_slots(self, state: GameState, player: int) -> List[int]:
        return [
            s
            for s in range(state.rules.hand_size)
            if state.knowledge[player][player][s] is None
        ]

    # ----------------------------------------------------------------- policy
    def choose(self, state: GameState, player: int) -> Move:
        phase = state.phase
        if phase is Phase.TURN_START:
            return self._turn_start(state, player)
        if phase is Phase.DECIDE_DRAWN:
            return self._decide_drawn(state, player)
        return self._knock_decision(state, player)

    def _turn_start(self, state: GameState, player: int) -> Move:
        legal = state.legal_moves()
        takes = [m for m in legal if m.kind is MoveKind.TAKE_DISCARD]
        if takes:
            top = state.discard_top
            assert top is not None
            slot, worst = self._worst_slot(state, player)
            if int(top) + self.p.take_discard_margin < worst:
                return Move(MoveKind.TAKE_DISCARD, slot)
        return Move(MoveKind.DRAW_DECK)

    def _decide_drawn(self, state: GameState, player: int) -> Move:
        legal = state.legal_moves()
        card = state.pending
        assert card is not None

        if card.is_number:
            slot, worst = self._worst_slot(state, player)
            replace = Move(MoveKind.REPLACE, slot)
            if int(card) + self.p.keep_margin < worst and replace in legal:
                return replace
            return Move(MoveKind.DISCARD_DRAWN)

        if Move(MoveKind.USE_DRAW2) in legal:
            # Two looks at the deck for free; always worth taking.
            return Move(MoveKind.USE_DRAW2)

        if card is Card.PEEK:
            unknown = self._unknown_slots(state, player)
            if unknown:
                # Peek the unknown slot we most expect to be a problem.
                est = self._estimates(state, player, player)
                target = max(unknown, key=lambda s: est[s])
                peek = Move(MoveKind.USE_PEEK, target)
                if peek in legal:
                    return peek
            return Move(MoveKind.DISCARD_DRAWN)

        if card is Card.SWAP:
            best = self._best_swap(state, player, legal)
            if best is not None:
                return best
            return Move(MoveKind.DISCARD_DRAWN)

        return Move(MoveKind.DISCARD_DRAWN)

    def _best_swap(
        self, state: GameState, player: int, legal: List[Move]
    ) -> Optional[Move]:
        swaps = [m for m in legal if m.kind is MoveKind.USE_SWAP]
        if not swaps:
            return None
        opp = other(player)
        mine = self._estimates(state, player, player)
        theirs = self._estimates(state, player, opp)
        # Gain is doubled: we shed points and hand them to the opponent, and the
        # margin is what the round is scored on.
        best = max(swaps, key=lambda m: mine[m.slot] - theirs[m.opp_slot])
        if mine[best.slot] - theirs[best.opp_slot] >= self.p.swap_margin:
            return best
        return None

    def _knock_decision(self, state: GameState, player: int) -> Move:
        legal = state.legal_moves()
        knock = Move(MoveKind.KNOCK)
        if knock not in legal:
            return Move(MoveKind.PASS_KNOCK)

        opp = other(player)
        my_total = sum(self._estimates(state, player, player))
        opp_total = sum(self._estimates(state, player, opp))
        unknown = len(self._unknown_slots(state, player))

        # The opponent gets one more turn after we call, which is worth a couple
        # of points to them on average; require a real cushion.
        if unknown <= self.p.knock_max_unknown_slots and my_total + self.p.knock_lead < opp_total:
            return Move(MoveKind.KNOCK)
        return Move(MoveKind.PASS_KNOCK)
