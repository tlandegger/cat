"""Belief tracking and determinization.

Two jobs:

1. **Card counting.** Everything a player has legitimately observed - their own
   known slots, every card ever discarded, opponent cards revealed by a take
   from the discard - is subtracted from the 54-card deck. What remains is the
   pool from which every hidden card must come.

2. **Provenance tilt.** Uniform-over-unseen is the right prior only for a slot
   nobody has acted on. A slot whose owner looked at a drawn card and *chose to
   keep it* is much more likely to be low, and a slot its owner has never seen
   is slightly more likely to be high (they would have replaced it if they knew).
   `SLOT_PRIORS` encodes that as multiplicative weights over card values.

`determinize` combines the two into a concrete world consistent with one
player's information, which is what the search agents sample over.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .cards import Card, DECK_COMPOSITION
from .state import NUM_PLAYERS, GameState, other


def _tilt(low_bias: float) -> Dict[Card, float]:
    """Weight vector over 0..9 plus power cards.

    ``low_bias`` > 0 pushes mass toward low cards, < 0 toward high cards. The
    shape is a simple geometric ramp, which is enough structure to matter
    without pretending to a precision we cannot justify.
    """
    weights = {Card(v): (1.0 + low_bias) ** (4.5 - v) for v in range(10)}
    for power in (Card.PEEK, Card.SWAP, Card.DRAW2):
        # Power cards cannot be chosen into a hand, so any slot the owner acted
        # on definitely is not one. Only untouched dealt slots can hold them,
        # and only when RuleConfig.deal_power_cards is on.
        weights[power] = 1.0
    return weights


#: Multiplicative priors keyed by the public provenance of a slot.
SLOT_PRIORS: Dict[str, Dict[Card, float]] = {
    # Never seen by its owner, never acted on: mild high-tilt, because a player
    # who knew it was low would have kept it and one who knew it was high would
    # have replaced it by now.
    "dealt": _tilt(-0.06),
    # Owner has seen it and left it there: they were content, so it skews low.
    "dealt_seen": _tilt(0.16),
    # Owner saw a drawn card and chose to keep it: strongly low.
    "kept_from_deck": _tilt(0.30),
    # Owner deliberately took a known card off the discard: strongly low.
    "took_discard": _tilt(0.30),
}
_UNIFORM = {c: 1.0 for c in DECK_COMPOSITION}


def seen_by(state: GameState, player: int) -> Counter:
    """Every card ``player`` has legitimately observed."""
    seen: Counter = Counter(state.discard)
    for owner in range(NUM_PLAYERS):
        for card in state.knowledge[player][owner]:
            if card is not None:
                seen[card] += 1
    if state.pending is not None and state.current == player:
        seen[state.pending] += 1
    return seen


def unseen_pool(state: GameState, player: int) -> Counter:
    """Multiset of cards whose location ``player`` does not know."""
    pool = Counter(DECK_COMPOSITION)
    pool.subtract(seen_by(state, player))
    bad = {c: n for c, n in pool.items() if n < 0}
    if bad:
        raise ValueError(f"belief inconsistency, over-counted {bad}")
    return +pool


def hidden_slots(state: GameState, player: int) -> List[Tuple[int, int]]:
    """``(owner, slot)`` pairs whose card ``player`` cannot name."""
    return [
        (owner, slot)
        for owner in range(NUM_PLAYERS)
        for slot in range(state.rules.hand_size)
        if state.knowledge[player][owner][slot] is None
    ]


def slot_distribution(
    state: GameState,
    player: int,
    owner: int,
    slot: int,
    use_provenance: bool = True,
) -> Dict[Card, float]:
    """Marginal probability distribution over the card in one hidden slot.

    Returns a point mass when ``player`` already knows the card.
    """
    known = state.knowledge[player][owner][slot]
    if known is not None:
        return {known: 1.0}

    pool = unseen_pool(state, player)
    prior = SLOT_PRIORS.get(state.provenance[owner][slot], _UNIFORM) if use_provenance else _UNIFORM
    weighted = {c: n * prior.get(c, 1.0) for c, n in pool.items() if n > 0}
    total = sum(weighted.values())
    if total <= 0:
        return {}
    return {c: w / total for c, w in weighted.items()}


def expected_slot_value(
    state: GameState,
    player: int,
    owner: int,
    slot: int,
    use_provenance: bool = True,
) -> float:
    """Expected point cost of one slot, under ``player``'s beliefs."""
    dist = slot_distribution(state, player, owner, slot, use_provenance)
    penalty = state.rules.power_card_score
    return sum(p * (int(c) if c.is_number else penalty) for c, p in dist.items())


def hand_estimates(
    state: GameState, player: int, owner: int, use_provenance: bool = True
) -> List[float]:
    """Expected point cost of every slot in ``owner``'s hand, in one pass.

    Equivalent to calling :func:`expected_slot_value` per slot, but computes the
    unseen pool once. This is the hot path during rollouts, so it matters.
    """
    pool = unseen_pool(state, player)
    penalty = state.rules.power_card_score
    items = [(c, n, (int(c) if c < 10 else penalty)) for c, n in pool.items() if n > 0]

    # Every unknown slot with the same provenance has the same expected value,
    # and there are only a handful of provenance labels, so compute each once
    # instead of re-walking the pool per slot.
    by_provenance: Dict[str, float] = {}

    def ev_for(label: str) -> float:
        cached = by_provenance.get(label)
        if cached is not None:
            return cached
        prior = SLOT_PRIORS.get(label, _UNIFORM) if use_provenance else _UNIFORM
        total = 0.0
        acc = 0.0
        for card, n, value in items:
            w = n * prior.get(card, 1.0)
            total += w
            acc += w * value
        result = acc / total if total > 0 else 0.0
        by_provenance[label] = result
        return result

    known_row = state.knowledge[player][owner]
    prov_row = state.provenance[owner]
    out: List[float] = []
    for slot in range(state.rules.hand_size):
        known = known_row[slot]
        if known is not None:
            out.append(float(int(known) if known < 10 else penalty))
        else:
            out.append(ev_for(prov_row[slot]))
    return out


def expected_hand_value(
    state: GameState, player: int, owner: int, use_provenance: bool = True
) -> float:
    return sum(hand_estimates(state, player, owner, use_provenance))


def _weighted_draw(pool: Counter, weights: Dict[Card, float], rng: random.Random) -> Card:
    """Draw one card from ``pool`` (mutating it) with the given weights."""
    items = [(c, n * weights.get(c, 1.0)) for c, n in pool.items() if n > 0]
    total = sum(w for _, w in items)
    if total <= 0:  # every candidate has zero weight; fall back to uniform
        items = [(c, float(n)) for c, n in pool.items() if n > 0]
        total = sum(w for _, w in items)
    target = rng.random() * total
    for card, weight in items:
        target -= weight
        if target <= 0:
            pool[card] -= 1
            return card
    card = items[-1][0]
    pool[card] -= 1
    return card


def determinize(
    state: GameState,
    player: int,
    rng: Optional[random.Random] = None,
    use_provenance: bool = True,
) -> GameState:
    """Sample a full-information world consistent with ``player``'s knowledge.

    Only ``state.knowledge[player]``, the discard pile and the public provenance
    table are read, so no ground truth leaks into the search.
    """
    rng = rng or random.Random()
    world = state.clone(rng=rng)
    pool = unseen_pool(state, player)

    # Fill hidden hand slots first, weighted by provenance, then the deck.
    unknown = hidden_slots(state, player)
    # Most-constrained first: slots with sharper priors should claim their cards
    # before the flat ones absorb them.
    unknown.sort(key=lambda os: state.provenance[os[0]][os[1]] == "dealt")
    for owner, slot in unknown:
        prior = SLOT_PRIORS.get(state.provenance[owner][slot], _UNIFORM) if use_provenance else _UNIFORM
        world.hands[owner][slot] = _weighted_draw(pool, prior, rng)

    if state.pending is not None and state.current != player:
        world.pending = _weighted_draw(pool, _UNIFORM, rng)

    remaining: List[Card] = []
    for card, n in pool.items():
        remaining.extend([card] * n)
    rng.shuffle(remaining)
    world.deck = remaining

    _rebuild_opponent_knowledge(world, player)
    return world


def _rebuild_opponent_knowledge(world: GameState, player: int) -> None:
    """Make the opponent's knowledge table agree with the determinized world.

    *Which* entries the opponent knows is public (you can see which slot someone
    peeks at, replaces or swaps), so preserving that structure leaks nothing.
    Only the values are refilled to match this sampled world.
    """
    opp = other(player)
    for owner in range(NUM_PLAYERS):
        for slot in range(world.rules.hand_size):
            if world.knowledge[opp][owner][slot] is not None:
                world.knowledge[opp][owner][slot] = world.hands[owner][slot]
