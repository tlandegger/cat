"""Advisor: parsing table-side descriptions and producing sane recommendations."""

from __future__ import annotations

import random

import pytest

from ratatatcat.advisor import (
    Observation,
    build_state,
    parse_hand,
    parse_pile,
    rank_moves,
)
from ratatatcat.agents import HeuristicAgent, ISMCTSAgent
from ratatatcat.cards import Card
from ratatatcat.cli import main, make_agent
from ratatatcat.state import MoveKind, Phase

from .test_engine import all_cards
from ratatatcat.cards import DECK_COMPOSITION


def test_parse_hand_reads_unknowns():
    assert parse_hand("3 ? ? 7") == [Card.N3, None, None, Card.N7]
    assert parse_hand("0,1,2,3") == [Card.N0, Card.N1, Card.N2, Card.N3]
    with pytest.raises(ValueError):
        parse_hand("1 2 3")


def test_parse_pile_handles_powers_and_blanks():
    assert parse_pile("") == []
    assert parse_pile("5 peek 2") == [Card.N5, Card.PEEK, Card.N2]


def test_build_state_matches_the_description():
    obs = Observation(
        my_hand=[Card.N3, None, None, Card.N7],
        opp_hand=[None, Card.N2, None, None],
        discard=[Card.N5, Card.N8],
        drawn=Card.N4,
    )
    state, me = build_state(obs, rng=random.Random(0))
    assert me == 0
    assert state.phase is Phase.DECIDE_DRAWN
    assert state.pending is Card.N4
    assert state.discard == [Card.N5, Card.N8]
    assert state.knowledge[me][0] == [Card.N3, None, None, Card.N7]
    assert state.knowledge[me][1] == [None, Card.N2, None, None]
    assert state.hands[0][0] is Card.N3 and state.hands[1][1] is Card.N2
    assert all_cards(state) == DECK_COMPOSITION


def test_build_state_without_a_drawn_card_starts_the_turn():
    state, _ = build_state(
        Observation(my_hand=[Card.N1, None, None, Card.N2], discard=[Card.N6]),
        rng=random.Random(1),
    )
    assert state.phase is Phase.TURN_START
    assert state.pending is None


def test_build_state_rejects_impossible_descriptions():
    obs = Observation(
        my_hand=[Card.N3, Card.N3, Card.N3, Card.N3],
        opp_hand=[Card.N3, None, None, None],  # a fifth 3
        discard=[],
    )
    with pytest.raises(ValueError):
        build_state(obs, rng=random.Random(2))


def test_rank_moves_returns_stats_sorted_best_first():
    state, me = build_state(
        Observation(
            my_hand=[Card.N9, None, None, Card.N1],
            discard=[Card.N5],
            drawn=Card.N0,
        ),
        rng=random.Random(3),
    )
    ranked = rank_moves(state, me, ISMCTSAgent(iterations=500, seed=1))
    assert ranked
    assert [r[1] for r in ranked] == sorted((r[1] for r in ranked), reverse=True)
    best = ranked[0][0]
    assert best.kind is MoveKind.REPLACE and best.slot == 0, "drop the known 9"


def test_advisor_recommends_taking_a_zero_off_the_discard():
    """Holding [8, ?, ?, 6] with a 0 on the discard, the 8 has to go.

    Unknown slots are worth about 6.2 here, so replacing the known 8 beats
    replacing an unknown by roughly 1.8 points of expected total. Run at the
    iteration count the ``advise`` command actually ships with -- below about
    2000 the search sometimes locks onto an unknown slot before it has explored
    the 8, which is the exploration limit documented in the README.
    """
    state, me = build_state(
        Observation(my_hand=[Card.N8, None, None, Card.N6], discard=[Card.N0]),
        rng=random.Random(4),
    )
    ranked = rank_moves(state, me, ISMCTSAgent(iterations=4000, seed=2))
    best = ranked[0][0]
    assert best.kind is MoveKind.TAKE_DISCARD
    assert best.slot == 0, "the known 8 is the worst card in the hand"


def test_make_agent_parses_specs():
    assert make_agent("ismcts:750").iterations == 750
    pimc = make_agent("pimc:30x2")
    assert (pimc.worlds, pimc.depth) == (30, 2)
    assert make_agent("heuristic").name == "heuristic"
    with pytest.raises(SystemExit):
        make_agent("nonesuch")


def test_advise_command_runs(capsys):
    code = main(
        [
            "advise",
            "--hand", "3 ? ? 7",
            "--discard", "5 8 2",
            "--drawn", "4",
            "--agent", "ismcts:200",
            "--seed", "1",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "recommended:" in out


def test_selfplay_command_runs(capsys):
    code = main(["selfplay", "--a", "heuristic", "--b", "random", "--seed", "5"])
    assert code == 0
    assert "final:" in capsys.readouterr().out


def test_bench_command_runs(capsys):
    code = main(
        ["bench", "--a", "heuristic", "--b", "random", "--rounds", "20", "--seed", "1"]
    )
    assert code == 0
    assert "win rate" in capsys.readouterr().out


def test_advise_rejects_an_impossible_position():
    with pytest.raises(SystemExit):
        main(["advise", "--hand", "3 3 3 3", "--discard", "3", "--agent", "heuristic"])
