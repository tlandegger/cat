"""Measure and calibrate knock-decision accuracy against ground truth.

This is the script behind the tuning constants in ``agents/search.py`` and the
"where the search goes wrong" section of the README. It exists so those numbers
can be re-derived rather than taken on trust.

**Ground truth for a position.** Hold the four real hands fixed, reshuffle only
the part of the deck nobody has seen, force the action, then play both seats
with the heuristic and average. That is the true value of the action for a
player whose future self plays the heuristic. It is an oracle an agent cannot
match -- it reads the real hidden cards -- so treat the "points lost" column as
regret against a clairvoyant, not as an achievable target. What matters is the
*comparison between agents* on the same positions.

Usage::

    python -m analysis.knock_calibration            # measure the defaults
    python -m analysis.knock_calibration --sweep    # sweep pv_weight
    python -m analysis.knock_calibration --heuristic-sweep
"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import random
import statistics
from typing import List, Tuple

from ratatatcat.agents import HeuristicAgent, ISMCTSAgent
from ratatatcat.agents.heuristic import HeuristicParams
from ratatatcat.agents.search import REWARD_SCALE, _playout
from ratatatcat.state import GameState, Move, MoveKind, Phase

KNOCK = Move(MoveKind.KNOCK)
PASS = Move(MoveKind.PASS_KNOCK)
HEURISTIC = HeuristicAgent()
CACHE = os.path.join(os.path.dirname(__file__), "knock_positions.pkl")

Scored = Tuple[GameState, int, float, float]


def denormalise(reward: float) -> float:
    """Undo the [0, 1] squashing so search values are comparable to points."""
    return (2.0 * reward - 1.0) * REWARD_SCALE


def true_value(state: GameState, player: int, move: Move, n: int, seed: int) -> float:
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        world = state.clone(rng=random.Random(rng.random()))
        rng.shuffle(world.deck)  # only the unseen deck order varies
        world.apply(move)
        _playout(world, HEURISTIC)
        total += world.utility(player)
    return total / n


def collect_positions(count: int, seed: int = 1) -> List[GameState]:
    """Sample knock decisions from heuristic self-play."""
    rng = random.Random(seed)
    out: List[GameState] = []
    while len(out) < count:
        state = GameState.deal(rng=rng)
        while not state.is_terminal:
            if state.phase is Phase.KNOCK_DECISION and KNOCK in state.legal_moves():
                out.append(state.clone(rng=random.Random(rng.random())))
                if len(out) >= count:
                    break
            state.apply(HEURISTIC.choose(state, state.current))
    return out


def score_positions(count: int, rollouts: int, use_cache: bool = True) -> List[Scored]:
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            data = pickle.load(fh)
        if len(data) >= count:
            return data[:count]

    positions = collect_positions(count)
    data: List[Scored] = []
    for i, state in enumerate(positions):
        player = state.current
        data.append(
            (
                state,
                player,
                true_value(state, player, KNOCK, rollouts, seed=i),
                true_value(state, player, PASS, rollouts, seed=i),
            )
        )
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(positions)}", flush=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh)
    return data


def baselines(data: List[Scored]) -> None:
    always = sum(abs(vk - vp) for _, _, vk, vp in data if vp > vk)
    never = sum(abs(vk - vp) for _, _, vk, vp in data if vk > vp)
    calls = sum(vk > vp for _, _, vk, vp in data)
    print(f"\n{len(data)} positions; ground truth calls in {calls}")
    print(f"  always call: {always:7.1f} points lost")
    print(f"  never call:  {never:7.1f}")
    print(f"  oracle:          0.0\n")


def evaluate(agent, data: List[Scored], label: str) -> None:
    calls = correct = 0
    lost = 0.0
    dk: List[float] = []
    dp: List[float] = []
    for state, player, vk, vp in data:
        move = agent.choose(state, player)
        stats = getattr(agent, "last_stats", {})
        if KNOCK in stats and PASS in stats and stats[KNOCK][0] and stats[PASS][0]:
            dk.append(denormalise(stats[KNOCK][1]) - vk)
            dp.append(denormalise(stats[PASS][1]) - vp)
        calls += move == KNOCK
        best = KNOCK if vk > vp else PASS
        if move == best:
            correct += 1
        else:
            lost += abs(vk - vp)
    line = f"  {label:26s} calls {calls:3d}/{len(data)}  correct {correct:3d}  lost {lost:7.1f}"
    if dk:
        line += (
            f"   bias: call {statistics.mean(dk):+5.2f}"
            f"  pass {statistics.mean(dp):+5.2f}"
        )
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=150)
    ap.add_argument("--rollouts", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--sweep", action="store_true", help="sweep ISMCTS pv_weight")
    ap.add_argument("--heuristic-sweep", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    data = score_positions(args.positions, args.rollouts, use_cache=not args.no_cache)
    baselines(data)

    if args.heuristic_sweep:
        print("heuristic knock rule:")
        rows = []
        for lead, unknown in itertools.product((-4, -2, 0, 2, 4, 6, 8), range(5)):
            params = HeuristicParams(
                knock_lead=lead, knock_max_unknown_slots=unknown
            )
            rows.append((lead, unknown, HeuristicAgent(params)))
        for lead, unknown, agent in rows:
            evaluate(agent, data, f"lead={lead:+.0f} maxUnknown={unknown}")
        return

    print("heuristic:")
    evaluate(HeuristicAgent(), data, "default params")

    print("\nISMCTS backups:")
    weights = (0.0, 0.3, 0.44, 0.6, 0.8, 1.0) if args.sweep else (0.0, 0.6)
    for weight in weights:
        agent = ISMCTSAgent(
            iterations=args.iterations, seed=3, backup="mix", pv_weight=weight
        )
        evaluate(agent, data, f"mix pv_weight={weight:.2f}")


if __name__ == "__main__":
    main()
