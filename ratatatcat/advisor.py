"""Turn a table-side description of a position into an analysable state.

At a real table you know: your own cards that you have seen, any opponent cards
you have seen, every card that has gone to the discard pile, and whose turn it
is. :func:`build_state` turns that into a :class:`GameState` whose *knowledge*
tables match your information exactly. Hidden cards are filled from the unseen
pool so the object is internally consistent; the solver never reads them, and
:func:`~ratatatcat.belief.determinize` resamples them every iteration anyway.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .agents.base import Agent
from .belief import unseen_pool
from .cards import Card, DECK_COMPOSITION, parse_card
from .rules import DEFAULT_RULES, RuleConfig
from .state import NUM_PLAYERS, GameState, Move, Phase


@dataclass
class Observation:
    """What one player can see. ``None`` in a hand slot means "unknown to me"."""

    my_hand: List[Optional[Card]]
    #: Opponent cards you happen to know (e.g. they took a 3 off the discard).
    opp_hand: List[Optional[Card]] = field(default_factory=lambda: [None] * 4)
    #: Every card discarded so far, oldest first. The last entry is the top.
    discard: List[Card] = field(default_factory=list)
    #: Card you have just drawn from the deck, if you are mid-turn.
    drawn: Optional[Card] = None
    #: How each slot came to be where it is; see ``belief.SLOT_PRIORS``.
    my_provenance: Optional[List[str]] = None
    opp_provenance: Optional[List[str]] = None
    #: Set if someone has already called.
    knocker: Optional[int] = None
    #: Turns you have already completed, used to gate the knock decision.
    my_turns_taken: int = 3
    opp_turns_taken: int = 3


def _default_provenance(hand: Sequence[Optional[Card]]) -> List[str]:
    """Assume seen slots were looked at and unseen ones never touched."""
    return ["dealt_seen" if c is not None else "dealt" for c in hand]


def build_state(
    obs: Observation,
    rules: RuleConfig = DEFAULT_RULES,
    rng: Optional[random.Random] = None,
) -> Tuple[GameState, int]:
    """Build a state from ``obs``. Returns ``(state, your_player_index)``.

    You are always player 0. Raises ``ValueError`` if the description is
    impossible (e.g. five 3s between your hand and the discard pile).
    """
    rng = rng or random.Random()
    me, opp = 0, 1
    if len(obs.my_hand) != rules.hand_size or len(obs.opp_hand) != rules.hand_size:
        raise ValueError(f"hands must have {rules.hand_size} slots")

    state = GameState(
        rules=rules,
        hands=[[Card.N0] * rules.hand_size for _ in range(NUM_PLAYERS)],
        deck=[],
        discard=list(obs.discard),
        knowledge=[
            [[None] * rules.hand_size for _ in range(NUM_PLAYERS)]
            for _ in range(NUM_PLAYERS)
        ],
        provenance=[
            obs.my_provenance or _default_provenance(obs.my_hand),
            obs.opp_provenance or _default_provenance(obs.opp_hand),
        ],
        current=me,
        phase=Phase.DECIDE_DRAWN if obs.drawn is not None else Phase.TURN_START,
        pending=obs.drawn,
        knocker=obs.knocker,
        turns_taken=[obs.my_turns_taken, obs.opp_turns_taken],
        rng=rng,
    )

    for slot, card in enumerate(obs.my_hand):
        if card is not None:
            state.knowledge[me][me][slot] = card
            state.hands[me][slot] = card
    for slot, card in enumerate(obs.opp_hand):
        if card is not None:
            state.knowledge[me][opp][slot] = card
            state.hands[opp][slot] = card

    # Fill what we cannot see from the unseen pool, so the object is a legal
    # 54-card world. unseen_pool validates the description on the way through.
    pool = unseen_pool(state, me)
    leftovers: List[Card] = []
    for card, n in pool.items():
        leftovers.extend([card] * n)
    rng.shuffle(leftovers)

    for owner, hand in ((me, obs.my_hand), (opp, obs.opp_hand)):
        for slot, card in enumerate(hand):
            if card is None:
                if not leftovers:
                    raise ValueError("not enough unseen cards to fill the hands")
                state.hands[owner][slot] = leftovers.pop()

    state.deck = leftovers

    # The opponent knows their own cards; which slots they know is public, and
    # by default we assume they know the two they were dealt face-down looks at
    # plus anything they chose into.
    for slot in range(rules.hand_size):
        if state.provenance[opp][slot] != "dealt":
            state.knowledge[opp][opp][slot] = state.hands[opp][slot]
        if state.provenance[me][slot] == "took_discard":
            state.knowledge[opp][me][slot] = state.hands[me][slot]

    return state, me


def rank_moves(
    state: GameState, player: int, agent: Agent
) -> List[Tuple[Move, int, float]]:
    """Run ``agent`` and return its move statistics, best first.

    Each entry is ``(move, samples, mean reward)``. Reward is on the agent's own
    scale: [0, 1] for ISMCTS, raw score margin for PIMC.
    """
    agent.choose(state, player)
    stats: Dict[Move, Tuple[int, float]] = getattr(agent, "last_stats", {})
    ranked = [(m, n, v) for m, (n, v) in stats.items()]
    ranked.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return ranked


def parse_hand(text: str, size: int = 4) -> List[Optional[Card]]:
    """Parse ``"3 ? ? 7"`` into ``[Card.N3, None, None, Card.N7]``."""
    tokens = text.replace(",", " ").split()
    if len(tokens) != size:
        raise ValueError(f"expected {size} slots, got {len(tokens)}")
    return [None if t in ("?", "-", "x") else parse_card(t) for t in tokens]


def parse_pile(text: str) -> List[Card]:
    text = text.strip()
    if not text:
        return []
    return [parse_card(t) for t in text.replace(",", " ").split()]
