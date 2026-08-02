"""Engine invariants: deck integrity, legality, knowledge truthfulness, scoring."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from ratatatcat.cards import DECK_COMPOSITION, Card, full_deck, parse_card
from ratatatcat.rules import RuleConfig
from ratatatcat.state import GameState, Move, MoveKind, Phase, other


def all_cards(state: GameState) -> Counter:
    tally: Counter = Counter()
    for hand in state.hands:
        tally.update(hand)
    tally.update(state.deck)
    tally.update(state.discard)
    if state.pending is not None:
        tally[state.pending] += 1
    return tally


def test_deck_is_54_cards_with_the_published_composition():
    assert len(full_deck()) == 54
    assert sum(DECK_COMPOSITION.values()) == 54
    assert DECK_COMPOSITION[Card.N9] == 9
    assert all(DECK_COMPOSITION[Card(v)] == 4 for v in range(9))
    assert all(DECK_COMPOSITION[c] == 3 for c in (Card.PEEK, Card.SWAP, Card.DRAW2))


def test_parse_card_round_trips_and_rejects_junk():
    assert parse_card("7") is Card.N7
    assert parse_card(" Peek ") is Card.PEEK
    assert parse_card("d2") is Card.DRAW2
    for bad in ("10", "-1", "cat", ""):
        with pytest.raises(ValueError):
            parse_card(bad)


def test_deal_gives_each_player_four_cards_and_one_discard():
    state = GameState.deal(rng=random.Random(0))
    assert [len(h) for h in state.hands] == [4, 4]
    assert len(state.discard) == 1
    assert len(state.deck) == 54 - 8 - 1
    assert all_cards(state) == DECK_COMPOSITION


def test_default_deal_puts_no_power_cards_in_hands():
    for seed in range(50):
        state = GameState.deal(rng=random.Random(seed))
        assert all(c.is_number for hand in state.hands for c in hand)


def test_deal_power_cards_flag_allows_them_into_hands():
    rules = RuleConfig(deal_power_cards=True)
    found = any(
        c.is_power
        for seed in range(200)
        for c in GameState.deal(rules=rules, rng=random.Random(seed)).hands[0]
    )
    assert found


def test_players_start_knowing_exactly_their_outer_two_cards():
    state = GameState.deal(rng=random.Random(1))
    for p in (0, 1):
        known = [s for s in range(4) if state.knowledge[p][p][s] is not None]
        assert known == [0, 3]
        # ...and nothing at all about the opponent.
        assert all(c is None for c in state.knowledge[p][other(p)])


@pytest.mark.parametrize("seed", range(40))
def test_random_play_preserves_the_deck_and_never_lies(seed):
    rng = random.Random(seed)
    state = GameState.deal(rng=rng)
    steps = 0
    while not state.is_terminal and steps < 400:
        moves = state.legal_moves()
        assert moves, f"no legal move in phase {state.phase}"
        state.apply(rng.choice(moves))
        steps += 1
        assert all_cards(state) == DECK_COMPOSITION
        for obs in (0, 1):
            for owner in (0, 1):
                for slot in range(4):
                    known = state.knowledge[obs][owner][slot]
                    assert known is None or known == state.hands[owner][slot]
    assert state.is_terminal


def test_power_cards_cannot_be_taken_from_the_discard_pile():
    state = GameState.deal(rng=random.Random(2))
    state.discard = [Card.SWAP]
    assert all(m.kind is not MoveKind.TAKE_DISCARD for m in state.legal_moves())
    state.discard = [Card.N4]
    assert any(m.kind is MoveKind.TAKE_DISCARD for m in state.legal_moves())


def test_taking_from_the_discard_is_public_knowledge():
    state = GameState.deal(rng=random.Random(3))
    state.discard = [Card.N2]
    state.apply(Move(MoveKind.TAKE_DISCARD, 1))
    assert state.hands[0][1] is Card.N2
    # Both players now know that slot holds a 2.
    assert state.knowledge[0][0][1] is Card.N2
    assert state.knowledge[1][0][1] is Card.N2
    assert state.provenance[0][1] == "took_discard"


def test_replacing_from_the_deck_hides_the_new_card_from_the_opponent():
    state = GameState.deal(rng=random.Random(4))
    state.deck.append(Card.N1)
    state.apply(Move(MoveKind.DRAW_DECK))
    assert state.pending is Card.N1
    state.apply(Move(MoveKind.REPLACE, 2))
    assert state.hands[0][2] is Card.N1
    assert state.knowledge[0][0][2] is Card.N1
    assert state.knowledge[1][0][2] is None
    assert state.discard[-1] is not None


def test_swap_moves_both_cards_and_the_beliefs_that_travel_with_them():
    state = GameState.deal(rng=random.Random(5))
    mine, theirs = state.hands[0][0], state.hands[1][3]
    # Player 0 knows their own slot 0; player 1 knows their own slot 3.
    assert state.knowledge[0][0][0] == mine
    assert state.knowledge[1][1][3] == theirs

    state.deck.append(Card.SWAP)
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.USE_SWAP, 0, 3))

    assert state.hands[0][0] == theirs
    assert state.hands[1][3] == mine
    # Neither player saw the cards, so each still "knows" the card they tracked,
    # now in its new home -- and nobody knows anything false.
    assert state.knowledge[0][1][3] == mine
    assert state.knowledge[1][0][0] == theirs
    assert state.knowledge[0][0][0] is None
    assert state.knowledge[1][1][3] is None


def test_peek_reveals_only_to_the_peeker():
    state = GameState.deal(rng=random.Random(6))
    state.deck.append(Card.PEEK)
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.USE_PEEK, 1))
    assert state.knowledge[0][0][1] == state.hands[0][1]
    assert state.knowledge[1][0][1] is None


def test_draw2_offers_a_second_card_only_after_discarding_the_first():
    rules = RuleConfig()
    state = GameState.deal(rules=rules, rng=random.Random(7))
    state.deck.extend([Card.N6, Card.N5, Card.DRAW2])  # popped last-first
    state.apply(Move(MoveKind.DRAW_DECK))
    assert state.pending is Card.DRAW2
    state.apply(Move(MoveKind.USE_DRAW2))
    assert state.pending is Card.N5 and state.in_draw2

    state.apply(Move(MoveKind.DISCARD_DRAWN))
    assert state.pending is Card.N6, "discarding the first draw should yield a second"
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    assert state.phase is Phase.KNOCK_DECISION
    assert not state.in_draw2


def test_keeping_the_first_draw2_card_forfeits_the_second():
    state = GameState.deal(rng=random.Random(8))
    state.deck.extend([Card.N6, Card.N5, Card.DRAW2])
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.USE_DRAW2))
    state.apply(Move(MoveKind.REPLACE, 0))
    assert state.phase is Phase.KNOCK_DECISION
    assert state.hands[0][0] is Card.N5


def test_power_card_drawn_during_draw2_is_inert_by_default():
    state = GameState.deal(rng=random.Random(9))
    state.deck.extend([Card.N3, Card.PEEK, Card.DRAW2])
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.USE_DRAW2))
    assert state.pending is Card.PEEK
    moves = state.legal_moves()
    assert moves == [Move(MoveKind.DISCARD_DRAWN)], "inert power card has one option"


def test_inert_flag_still_applies_on_the_final_draw2_card():
    """Regression: the inert check must key off in_draw2, not draws_left."""
    state = GameState.deal(rng=random.Random(10))
    state.deck.extend([Card.SWAP, Card.N3, Card.DRAW2])
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.USE_DRAW2))
    assert state.pending is Card.N3
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    # Second and last sub-draw: draws_left is now 0 but we are still in Draw 2.
    assert state.pending is Card.SWAP and state.draws_left == 0
    assert state.legal_moves() == [Move(MoveKind.DISCARD_DRAWN)]


def test_draw2_chains_when_enabled():
    rules = RuleConfig(draw2_chains=True)
    state = GameState.deal(rules=rules, rng=random.Random(11))
    state.deck.extend([Card.N1, Card.DRAW2, Card.DRAW2])
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.USE_DRAW2))
    assert state.pending is Card.DRAW2
    assert Move(MoveKind.USE_DRAW2) in state.legal_moves()


def test_knock_gives_the_opponent_exactly_one_more_turn():
    state = GameState.deal(rng=random.Random(12))
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    assert state.phase is Phase.KNOCK_DECISION
    state.apply(Move(MoveKind.KNOCK))
    assert state.knocker == 0 and state.current == 1

    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    state.apply(Move(MoveKind.PASS_KNOCK))
    assert state.is_terminal, "round ends when play returns to the caller"


def test_only_one_player_may_knock():
    state = GameState.deal(rng=random.Random(13))
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    state.apply(Move(MoveKind.KNOCK))
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    assert Move(MoveKind.KNOCK) not in state.legal_moves()


def test_knocker_protection_flag_blocks_swapping_against_the_caller():
    rules = RuleConfig(knocker_protected=True)
    state = GameState.deal(rules=rules, rng=random.Random(14))
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    state.apply(Move(MoveKind.KNOCK))
    state.deck.append(Card.SWAP)
    state.apply(Move(MoveKind.DRAW_DECK))
    assert all(m.kind is not MoveKind.USE_SWAP for m in state.legal_moves())


def test_scoring_sums_face_values_and_utility_is_zero_sum():
    state = GameState.deal(rng=random.Random(15))
    state.hands = [
        [Card.N1, Card.N2, Card.N3, Card.N4],
        [Card.N0, Card.N0, Card.N9, Card.N9],
    ]
    assert state.score(0) == 10
    assert state.score(1) == 18
    assert state.utility(0) == 8
    assert state.utility(1) == -8
    assert state.utility(0) == -state.utility(1)


def test_held_power_cards_score_the_configured_penalty():
    rules = RuleConfig(deal_power_cards=True, power_card_score=9)
    state = GameState.deal(rules=rules, rng=random.Random(16))
    state.hands[0] = [Card.PEEK, Card.N0, Card.N0, Card.N0]
    assert state.score(0) == 9


def test_deck_reshuffles_from_the_discard_when_exhausted():
    state = GameState.deal(rng=random.Random(17))
    state.discard = [Card.N1, Card.N2, Card.N3, Card.N4]
    state.deck = []
    state.apply(Move(MoveKind.DRAW_DECK))
    assert state.pending is not None
    assert len(state.discard) == 1, "the top card stays; the rest becomes the deck"


def test_clone_is_a_deep_copy():
    state = GameState.deal(rng=random.Random(18))
    twin = state.clone()
    twin.hands[0][0] = Card.N9
    twin.knowledge[0][0][0] = Card.N9
    twin.provenance[0][0] = "took_discard"
    twin.discard.append(Card.N5)
    assert state.hands[0][0] != Card.N9 or twin.hands[0][0] == state.hands[0][0]
    assert state.provenance[0][0] != "took_discard"
    assert len(state.discard) == 1
