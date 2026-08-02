"""Agent behaviour: legality, information hygiene, and that search actually helps."""

from __future__ import annotations

import random

import pytest

from ratatatcat.agents import (
    HeuristicAgent,
    ISMCTSAgent,
    NeverKnockRandomAgent,
    PIMCAgent,
    RandomAgent,
)
from ratatatcat.agents.search import candidate_moves
from ratatatcat.cards import Card
from ratatatcat.simulate import play_match, play_round
from ratatatcat.state import GameState, Move, MoveKind, Phase

ALL_AGENTS = [
    lambda: RandomAgent(seed=0),
    lambda: NeverKnockRandomAgent(seed=0),
    lambda: HeuristicAgent(),
    lambda: PIMCAgent(worlds=4, depth=1, seed=0),
    lambda: ISMCTSAgent(iterations=40, seed=0),
]


@pytest.mark.parametrize("make", ALL_AGENTS)
def test_agents_only_ever_return_legal_moves(make):
    agent = make()
    rng = random.Random(1)
    state = GameState.deal(rng=rng)
    steps = 0
    while not state.is_terminal and steps < 120:
        move = agent.choose(state, state.current)
        assert move in state.legal_moves(), f"{agent.name} returned an illegal move"
        state.apply(move)
        steps += 1


@pytest.mark.parametrize("make", ALL_AGENTS)
def test_a_full_round_terminates_for_every_agent(make):
    state = play_round(make(), make(), rng=random.Random(2))
    assert state.is_terminal
    assert state.score(0) >= 0 and state.score(1) >= 0


def test_heuristic_takes_a_low_discard_over_a_known_high_card():
    state = GameState.deal(rng=random.Random(3))
    state.hands[0] = [Card.N8, Card.N2, Card.N2, Card.N2]
    state.knowledge[0][0] = [Card.N8, Card.N2, Card.N2, Card.N2]
    state.discard = [Card.N0]
    move = HeuristicAgent().choose(state, 0)
    assert move == Move(MoveKind.TAKE_DISCARD, 0), "should dump the 8 for a 0"


def test_heuristic_declines_a_discard_worse_than_everything_it_holds():
    state = GameState.deal(rng=random.Random(4))
    state.hands[0] = [Card.N1, Card.N0, Card.N1, Card.N0]
    state.knowledge[0][0] = [Card.N1, Card.N0, Card.N1, Card.N0]
    state.discard = [Card.N9]
    assert HeuristicAgent().choose(state, 0) == Move(MoveKind.DRAW_DECK)


def test_heuristic_keeps_a_good_draw_and_discards_a_bad_one():
    state = GameState.deal(rng=random.Random(5))
    state.hands[0] = [Card.N7, Card.N1, Card.N1, Card.N1]
    state.knowledge[0][0] = [Card.N7, Card.N1, Card.N1, Card.N1]

    state.deck.append(Card.N0)
    state.apply(Move(MoveKind.DRAW_DECK))
    assert HeuristicAgent().choose(state, 0) == Move(MoveKind.REPLACE, 0)

    state = GameState.deal(rng=random.Random(5))
    state.hands[0] = [Card.N2, Card.N1, Card.N1, Card.N1]
    state.knowledge[0][0] = [Card.N2, Card.N1, Card.N1, Card.N1]
    state.deck.append(Card.N9)
    state.apply(Move(MoveKind.DRAW_DECK))
    assert HeuristicAgent().choose(state, 0) == Move(MoveKind.DISCARD_DRAWN)


def test_heuristic_peeks_at_a_slot_it_cannot_see():
    state = GameState.deal(rng=random.Random(6))
    state.deck.append(Card.PEEK)
    state.apply(Move(MoveKind.DRAW_DECK))
    move = HeuristicAgent().choose(state, 0)
    assert move.kind is MoveKind.USE_PEEK
    assert state.knowledge[0][0][move.slot] is None, "peek a slot we don't know"


def test_heuristic_swaps_its_worst_card_away():
    state = GameState.deal(rng=random.Random(7))
    state.hands[0] = [Card.N9, Card.N0, Card.N0, Card.N0]
    state.knowledge[0][0] = [Card.N9, Card.N0, Card.N0, Card.N0]
    state.knowledge[0][1] = [Card.N1, None, None, None]
    state.deck.append(Card.SWAP)
    state.apply(Move(MoveKind.DRAW_DECK))
    move = HeuristicAgent().choose(state, 0)
    assert move.kind is MoveKind.USE_SWAP
    assert move.slot == 0, "give away the 9"
    assert move.opp_slot == 0, "take the known 1"


def test_heuristic_calls_when_far_ahead_and_not_when_behind():
    def position(my_cards):
        state = GameState.deal(rng=random.Random(8))
        state.hands[0] = list(my_cards)
        state.knowledge[0][0] = list(my_cards)
        state.knowledge[0][1] = [Card.N9, Card.N9, Card.N9, Card.N9]
        state.hands[1] = [Card.N9] * 4
        state.turns_taken = [4, 4]
        state.phase = Phase.KNOCK_DECISION
        return state

    ahead = position([Card.N0, Card.N0, Card.N1, Card.N1])
    assert HeuristicAgent().choose(ahead, 0) == Move(MoveKind.KNOCK)

    behind = position([Card.N9, Card.N9, Card.N9, Card.N8])
    assert HeuristicAgent().choose(behind, 0) == Move(MoveKind.PASS_KNOCK)


def test_agents_do_not_read_the_hidden_hand():
    """Scramble cards the agent cannot see; its decision must not change.

    This is the information-hygiene check: an agent that peeks at
    ``state.hands`` for slots it has no knowledge of will fail here.
    """
    for make in (lambda: HeuristicAgent(), lambda: ISMCTSAgent(iterations=200, seed=3)):
        rng = random.Random(9)
        state = GameState.deal(rng=rng)
        for _ in range(6):
            state.apply(rng.choice(state.legal_moves()))
        if state.is_terminal or state.current != 0:
            continue

        baseline = make().choose(state, 0)
        altered = state.clone(rng=random.Random(9))
        # Swap two of the opponent's unknown slots and reshuffle the deck.
        unknown = [s for s in range(4) if state.knowledge[0][1][s] is None]
        if len(unknown) >= 2:
            a, b = unknown[0], unknown[1]
            altered.hands[1][a], altered.hands[1][b] = (
                altered.hands[1][b],
                altered.hands[1][a],
            )
        random.Random(99).shuffle(altered.deck)
        assert make().choose(altered, 0) == baseline


def test_ismcts_reports_statistics_for_every_root_move():
    state = GameState.deal(rng=random.Random(10))
    agent = ISMCTSAgent(iterations=150, seed=1)
    move = agent.choose(state, 0)
    assert move in agent.last_stats
    assert set(agent.last_stats) <= set(state.legal_moves())
    assert sum(n for n, _ in agent.last_stats.values()) > 0


def test_ismcts_prefers_the_obviously_right_replacement():
    """Holding a known 9, drawing a 0: the only sane move is to swap them."""
    state = GameState.deal(rng=random.Random(11))
    state.hands[0] = [Card.N9, Card.N1, Card.N1, Card.N1]
    state.knowledge[0][0] = [Card.N9, Card.N1, Card.N1, Card.N1]
    state.deck.append(Card.N0)
    state.apply(Move(MoveKind.DRAW_DECK))
    assert ISMCTSAgent(iterations=600, seed=2).choose(state, 0) == Move(
        MoveKind.REPLACE, 0
    )


def test_candidate_moves_keeps_the_worst_slot_as_a_replacement_target():
    """Pruning must never drop the slot it is most profitable to replace."""
    state = GameState.deal(rng=random.Random(20))
    state.hands[0] = [Card.N9, Card.N0, Card.N1, Card.N2]
    state.knowledge[0][0] = [Card.N9, Card.N0, Card.N1, Card.N2]
    state.deck.append(Card.N3)
    state.apply(Move(MoveKind.DRAW_DECK))

    pruned = candidate_moves(state, 0, keep=2)
    assert Move(MoveKind.REPLACE, 0) in pruned, "the 9 must stay a candidate"
    assert Move(MoveKind.DISCARD_DRAWN) in pruned
    assert Move(MoveKind.REPLACE, 1) not in pruned, "replacing the 0 is dominated"
    assert set(pruned) <= set(state.legal_moves())


def test_candidate_moves_cuts_swap_branching():
    state = GameState.deal(rng=random.Random(21))
    state.deck.append(Card.SWAP)
    state.apply(Move(MoveKind.DRAW_DECK))
    legal = state.legal_moves()
    pruned = candidate_moves(state, 0, keep=2)
    assert sum(m.kind is MoveKind.USE_SWAP for m in legal) == 16
    assert sum(m.kind is MoveKind.USE_SWAP for m in pruned) == 2
    assert set(pruned) <= set(legal)


def test_candidate_moves_never_prunes_a_two_option_decision():
    """Knock decisions have two options and must both survive."""
    state = GameState.deal(rng=random.Random(22))
    state.apply(Move(MoveKind.DRAW_DECK))
    state.apply(Move(MoveKind.DISCARD_DRAWN))
    assert state.phase is Phase.KNOCK_DECISION
    assert set(candidate_moves(state, 0)) == set(state.legal_moves())


def test_backup_mode_is_validated():
    with pytest.raises(ValueError):
        ISMCTSAgent(backup="nonsense")
    for mode in ("mean", "pv", "mix"):
        ISMCTSAgent(iterations=20, backup=mode, seed=0)


@pytest.mark.parametrize("mode", ["mean", "pv", "mix"])
def test_every_backup_mode_plays_legally(mode):
    agent = ISMCTSAgent(iterations=60, seed=0, backup=mode)
    state = GameState.deal(rng=random.Random(23))
    steps = 0
    while not state.is_terminal and steps < 80:
        move = agent.choose(state, state.current)
        assert move in state.legal_moves()
        state.apply(move)
        steps += 1


def test_pruning_can_be_disabled():
    state = GameState.deal(rng=random.Random(24))
    state.deck.append(Card.SWAP)
    state.apply(Move(MoveKind.DRAW_DECK))
    agent = ISMCTSAgent(iterations=120, seed=0, prune=False)
    agent.choose(state, 0)
    assert len(agent.last_stats) == len(state.legal_moves())


@pytest.mark.slow
def test_heuristic_comfortably_beats_random():
    result = play_match(HeuristicAgent(), RandomAgent(seed=1), rounds=300, seed=7)
    assert result.win_rate_a > 0.70, result.summary("heuristic", "random")


@pytest.mark.slow
def test_ismcts_is_at_least_competitive_with_the_heuristic():
    """The search must not be *worse* than the policy it rolls out with.

    Deliberately not asserting that ISMCTS beats the heuristic: measured over
    500 rounds its edge is inside one standard error, so such a test would be
    encoding a claim the data does not support and would flake. What is worth
    guarding is the regression direction -- the naive configurations lost by
    1.2 to 1.9 points, so a margin below -1.0 means a real bug has come back.
    """
    result = play_match(
        ISMCTSAgent(iterations=1500, seed=3), HeuristicAgent(), rounds=150, seed=13
    )
    assert result.mean_margin > -1.0, result.summary("ismcts", "heuristic")
    assert result.win_rate_a > 0.45, result.summary("ismcts", "heuristic")


@pytest.mark.slow
def test_ismcts_comfortably_beats_random():
    """A sanity floor: whatever else is true, it must crush a random player."""
    result = play_match(
        ISMCTSAgent(iterations=600, seed=3), RandomAgent(seed=1), rounds=150, seed=17
    )
    assert result.win_rate_a > 0.65, result.summary("ismcts", "random")
