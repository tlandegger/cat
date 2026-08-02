"""Command line entry point.

    python -m ratatatcat play      # play a round against the solver
    python -m ratatatcat advise    # analyse a position from the table
    python -m ratatatcat bench     # agent vs agent win rates
    python -m ratatatcat selfplay  # watch two agents play, move by move
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import List, Optional

from .advisor import Observation, build_state, parse_hand, parse_pile, rank_moves
from .agents import REGISTRY, HeuristicAgent, ISMCTSAgent
from .agents.base import Agent
from .cards import Card, parse_card
from .rules import DEFAULT_RULES, RuleConfig
from .simulate import play_match, play_round
from .state import GameState, Move, MoveKind, Phase, other


def make_agent(spec: str, seed: Optional[int] = None) -> Agent:
    """Build an agent from a spec like ``ismcts:4000`` or ``pimc:80x3``."""
    name, _, arg = spec.partition(":")
    if name not in REGISTRY:
        raise SystemExit(f"unknown agent {name!r}; choose from {', '.join(REGISTRY)}")
    cls = REGISTRY[name]
    if name == "ismcts":
        return cls(iterations=int(arg) if arg else 2000, seed=seed)
    if name == "pimc":
        worlds, _, depth = arg.partition("x")
        return cls(
            worlds=int(worlds) if worlds else 120,
            depth=int(depth) if depth else 4,
            seed=seed,
        )
    if name in ("random", "random-nocall"):
        return cls(seed=seed)
    return cls()


# ----------------------------------------------------------------------- play
def _describe(state: GameState, viewpoint: int) -> str:
    return state.render(viewpoint)


def cmd_play(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    bot = make_agent(args.opponent, seed=args.seed)
    state = GameState.deal(rules=RuleConfig(), rng=rng, first_player=0)
    you, them = 0, 1

    print("You are player 0. Slots are numbered 0-3; you were dealt a look at 0 and 3.")
    print(f"Your visible cards: {_describe(state, you)}\n")

    while not state.is_terminal:
        if state.current == you:
            print("-" * 58)
            print(f"Your view:\n{_describe(state, you)}")
            if state.pending is not None:
                print(f"  you drew: {state.pending}")
            move = _prompt_move(state, you, args.hint)
        else:
            move = bot.choose(state, them)
            print(f"  opponent: {move}")
        state.apply(move)
        if state.knocker is not None and state.phase is not Phase.REVEALED:
            print(f"  *** player {state.knocker} called Rat-a-Tat Cat ***")

    print("=" * 58)
    print("Final hands:")
    print(state.render(None))
    ys, ts = state.score(you), state.score(them)
    print(f"\n  you: {ys}   opponent: {ts}")
    print("  " + ("you win!" if ys < ts else "opponent wins" if ts < ys else "tie"))
    return 0


def _prompt_move(state: GameState, player: int, hint: bool) -> Move:
    legal = state.legal_moves()
    if len(legal) == 1:
        print(f"  (forced) {legal[0]}")
        return legal[0]
    if hint:
        advisor = ISMCTSAgent(iterations=1500)
        best = advisor.choose(state, player)
        print(f"  hint: {best}")
    for i, m in enumerate(legal):
        print(f"   [{i}] {m}")
    while True:
        raw = input("  choose> ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(legal):
            return legal[int(raw)]
        print("  enter one of the listed numbers")


# --------------------------------------------------------------------- advise
def cmd_advise(args: argparse.Namespace) -> int:
    try:
        my_hand = parse_hand(args.hand)
        opp_hand = parse_hand(args.opp) if args.opp else [None] * 4
        discard = parse_pile(args.discard)
        drawn = parse_card(args.drawn) if args.drawn else None
    except ValueError as exc:
        raise SystemExit(f"could not read the position: {exc}")

    obs = Observation(
        my_hand=my_hand,
        opp_hand=opp_hand,
        discard=discard,
        drawn=drawn,
        knocker=args.knocker,
    )
    try:
        state, me = build_state(obs, rng=random.Random(args.seed))
    except ValueError as exc:
        raise SystemExit(f"that position is not possible: {exc}")

    agent = make_agent(args.agent, seed=args.seed)
    print("Position:")
    print(state.render(me))
    if drawn is not None:
        print(f"  you drew: {drawn}")
    print(f"\nAnalysing with {args.agent} ...\n")

    ranked = rank_moves(state, me, agent)
    if not ranked:
        print("  no choice to make here")
        return 0
    width = max(len(str(m)) for m, _, _ in ranked)
    print(f"  {'move'.ljust(width)}   samples   value")
    for move, n, value in ranked:
        print(f"  {str(move).ljust(width)}   {n:7d}   {value:6.3f}")
    print(f"\n  recommended: {ranked[0][0]}")
    return 0


# ---------------------------------------------------------------------- bench
def cmd_bench(args: argparse.Namespace) -> int:
    a = make_agent(args.a, seed=args.seed)
    b = make_agent(args.b, seed=None if args.seed is None else args.seed + 1)
    print(f"playing {args.rounds} rounds: {args.a} vs {args.b}", flush=True)
    result = play_match(
        a, b, rounds=args.rounds, seed=args.seed, progress=args.progress
    )
    print()
    print(result.summary(args.a, args.b))
    return 0


# ------------------------------------------------------------------- selfplay
def cmd_selfplay(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    a = make_agent(args.a, seed=args.seed)
    b = make_agent(args.b, seed=None if args.seed is None else args.seed + 1)
    state = GameState.deal(rng=rng)
    agents = {0: a, 1: b}
    names = {0: args.a, 1: args.b}

    print(f"P0 = {args.a}   P1 = {args.b}")
    print(state.render(None), "\n")
    while not state.is_terminal:
        p = state.current
        move = agents[p].choose(state, p)
        pending = f" (drew {state.pending})" if state.pending is not None else ""
        print(f"  P{p} [{names[p]}]{pending}: {move}")
        state.apply(move)
    print("\nfinal:")
    print(state.render(None))
    print(f"  P0 {state.score(0)}  |  P1 {state.score(1)}  |  caller: {state.knocker}")
    return 0


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ratatatcat", description="Rat-a-Tat Cat engine and solver (2 players)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("play", help="play a round against a bot")
    pl.add_argument("--opponent", default="ismcts:2000")
    pl.add_argument("--hint", action="store_true", help="show the solver's pick")
    pl.add_argument("--seed", type=int)
    pl.set_defaults(func=cmd_play)

    ad = sub.add_parser("advise", help="analyse a position you describe")
    ad.add_argument("--hand", required=True, help='your 4 slots, e.g. "3 ? ? 7"')
    ad.add_argument("--opp", help='opponent slots you know, e.g. "? ? 2 ?"')
    ad.add_argument("--discard", default="", help="discard pile, oldest first")
    ad.add_argument("--drawn", help="card you just drew from the deck")
    ad.add_argument("--knocker", type=int, choices=[0, 1], help="who has called")
    ad.add_argument("--agent", default="ismcts:4000")
    ad.add_argument("--seed", type=int)
    ad.set_defaults(func=cmd_advise)

    bn = sub.add_parser("bench", help="win rates between two agents")
    bn.add_argument("--a", default="ismcts:1000")
    bn.add_argument("--b", default="heuristic")
    bn.add_argument("--rounds", type=int, default=200)
    bn.add_argument("--seed", type=int, default=0)
    bn.add_argument("--progress", action="store_true")
    bn.set_defaults(func=cmd_bench)

    sp = sub.add_parser("selfplay", help="watch two agents play one round")
    sp.add_argument("--a", default="ismcts:1000")
    sp.add_argument("--b", default="heuristic")
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_selfplay)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
