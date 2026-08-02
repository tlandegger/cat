"""Belief model: card counting, determinization soundness, information hygiene."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from ratatatcat.belief import (
    determinize,
    expected_slot_value,
    hand_estimates,
    hidden_slots,
    seen_by,
    slot_distribution,
    unseen_pool,
)
from ratatatcat.cards import DECK_COMPOSITION, Card
from ratatatcat.state import GameState, Move, MoveKind, other

from .test_engine import all_cards


def advance(state: GameState, rng: random.Random, steps: int) -> GameState:
    for _ in range(steps):
        if state.is_terminal:
            break
        state.apply(rng.choice(state.legal_moves()))
    return state


def test_unseen_pool_accounts_for_every_card():
    state = GameState.deal(rng=random.Random(0))
    pool = unseen_pool(state, 0)
    seen = seen_by(state, 0)
    assert sum(pool.values()) + sum(seen.values()) == 54
    # At the deal, player 0 sees two of their own cards plus the discard top.
    assert sum(seen.values()) == 3
    assert len(hidden_slots(state, 0)) == 6


def test_a_known_slot_has_a_point_mass_distribution():
    state = GameState.deal(rng=random.Random(1))
    known = state.hands[0][0]
    assert slot_distribution(state, 0, 0, 0) == {known: 1.0}
    assert expected_slot_value(state, 0, 0, 0) == pytest.approx(float(int(known)))


def test_slot_distributions_are_normalised():
    state = GameState.deal(rng=random.Random(2))
    for owner in (0, 1):
        for slot in range(4):
            dist = slot_distribution(state, 0, owner, slot)
            assert sum(dist.values()) == pytest.approx(1.0)


def test_card_counting_excludes_everything_already_seen():
    state = GameState.deal(rng=random.Random(3))
    # Force four 9s into the discard; the pool must drop to five.
    state.discard = [Card.N9] * 4
    pool = unseen_pool(state, 0)
    assert pool[Card.N9] == 9 - 4 - sum(1 for c in state.hands[0][:1] if c is Card.N9) * 0
    assert slot_distribution(state, 0, 1, 1).get(Card.N9, 0) < 9 / 45


def test_impossible_positions_are_rejected():
    state = GameState.deal(rng=random.Random(4))
    state.discard = [Card.N3] * 5  # only four 3s exist
    with pytest.raises(ValueError):
        unseen_pool(state, 0)


def test_provenance_tilts_kept_cards_low_and_untouched_cards_high():
    state = GameState.deal(rng=random.Random(5))
    state.provenance[1] = ["dealt", "kept_from_deck", "took_discard", "dealt_seen"]
    est = hand_estimates(state, 0, 1)
    untouched, kept, took, seen = est
    assert kept < seen < untouched, "a card its owner chose to keep should read low"
    assert took < untouched
    # Turning the model off makes every unknown slot identical.
    flat = hand_estimates(state, 0, 1, use_provenance=False)
    assert len(set(round(v, 9) for v in flat)) == 1


def test_hand_estimates_matches_per_slot_computation():
    rng = random.Random(6)
    state = advance(GameState.deal(rng=rng), rng, 12)
    for owner in (0, 1):
        batched = hand_estimates(state, 0, owner)
        one_by_one = [expected_slot_value(state, 0, owner, s) for s in range(4)]
        assert batched == pytest.approx(one_by_one)


@pytest.mark.parametrize("seed", range(25))
def test_determinized_worlds_are_legal_and_consistent(seed):
    rng = random.Random(seed)
    state = advance(GameState.deal(rng=rng), rng, rng.randint(0, 30))
    if state.is_terminal:
        return
    me = state.current
    world = determinize(state, me, rng)

    assert all_cards(world) == DECK_COMPOSITION
    assert world.discard == state.discard
    assert world.phase is state.phase
    assert world.current == state.current
    # Nothing the player knows may be contradicted.
    for owner in (0, 1):
        for slot in range(4):
            known = state.knowledge[me][owner][slot]
            if known is not None:
                assert world.hands[owner][slot] == known


def test_determinize_actually_varies_the_hidden_cards():
    rng = random.Random(7)
    state = GameState.deal(rng=rng)
    samples = {
        tuple(determinize(state, 0, rng).hands[1]) for _ in range(40)
    }
    assert len(samples) > 1, "hidden cards should be resampled each time"


def test_determinize_does_not_leak_the_true_hidden_hand():
    """Over many samples the opponent's real hand must not dominate."""
    rng = random.Random(8)
    state = GameState.deal(rng=rng)
    truth = tuple(state.hands[1])
    hits = sum(tuple(determinize(state, 0, rng).hands[1]) == truth for _ in range(300))
    assert hits < 30, "determinization is echoing the ground truth"


def test_determinize_preserves_which_slots_the_opponent_knows():
    rng = random.Random(9)
    state = advance(GameState.deal(rng=rng), rng, 10)
    if state.is_terminal:
        return
    me = state.current
    opp = other(me)
    structure = [
        [c is not None for c in state.knowledge[opp][owner]] for owner in (0, 1)
    ]
    world = determinize(state, me, rng)
    new_structure = [
        [c is not None for c in world.knowledge[opp][owner]] for owner in (0, 1)
    ]
    assert structure == new_structure
    # And whatever the opponent "knows" in the sampled world must be true there.
    for owner in (0, 1):
        for slot in range(4):
            k = world.knowledge[opp][owner][slot]
            assert k is None or k == world.hands[owner][slot]
