"""Head-to-head evaluation of agents.

Seats are swapped every other round so that any first-player advantage cancels
out, and both agents see the same deals.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .agents.base import Agent
from .rules import DEFAULT_RULES, RuleConfig
from .state import GameState, Phase


@dataclass
class MatchResult:
    rounds: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    margins: List[float] = field(default_factory=list)
    scores_a: List[int] = field(default_factory=list)
    scores_b: List[int] = field(default_factory=list)
    knocks_a: int = 0
    knocks_b: int = 0

    @property
    def win_rate_a(self) -> float:
        """Wins for A as a fraction, counting draws as half."""
        if not self.rounds:
            return 0.0
        return (self.wins_a + 0.5 * self.draws) / self.rounds

    @property
    def mean_margin(self) -> float:
        return sum(self.margins) / len(self.margins) if self.margins else 0.0

    @property
    def margin_stderr(self) -> float:
        n = len(self.margins)
        if n < 2:
            return 0.0
        mean = self.mean_margin
        var = sum((m - mean) ** 2 for m in self.margins) / (n - 1)
        return math.sqrt(var / n)

    @property
    def win_rate_stderr(self) -> float:
        """Standard error of ``win_rate_a`` under a binomial model."""
        if self.rounds < 2:
            return 0.0
        p = self.win_rate_a
        return math.sqrt(max(p * (1 - p), 0.0) / self.rounds)

    def summary(self, name_a: str, name_b: str) -> str:
        return (
            f"{name_a} vs {name_b} over {self.rounds} rounds\n"
            f"  win rate ({name_a}): {self.win_rate_a:6.1%} +/- {self.win_rate_stderr:.1%}"
            f"   [{self.wins_a}W {self.wins_b}L {self.draws}D]\n"
            f"  mean margin:        {self.mean_margin:+6.2f} +/- {self.margin_stderr:.2f}"
            f"   (points, {name_a}'s favour)\n"
            f"  mean score:         {name_a} {_mean(self.scores_a):.2f}"
            f" | {name_b} {_mean(self.scores_b):.2f}\n"
            f"  calls made:         {name_a} {self.knocks_a} | {name_b} {self.knocks_b}"
        )


def _mean(xs: Sequence[int]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def play_round(
    agent_a: Agent,
    agent_b: Agent,
    rules: RuleConfig = DEFAULT_RULES,
    rng: Optional[random.Random] = None,
    seat_a: int = 0,
    first_player: int = 0,
) -> GameState:
    """Play one round. ``seat_a`` says which seat index agent_a occupies."""
    rng = rng or random.Random()
    state = GameState.deal(rules=rules, rng=rng, first_player=first_player)
    agent_a.reset()
    agent_b.reset()
    by_seat = {seat_a: agent_a, 1 - seat_a: agent_b}

    guard = 0
    while not state.is_terminal and guard < rules.max_turns * 8:
        agent = by_seat[state.current]
        move = agent.choose(state, state.current)
        state.apply(move)
        guard += 1
    if not state.is_terminal:
        state.phase = Phase.REVEALED
    return state


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    rounds: int = 200,
    rules: RuleConfig = DEFAULT_RULES,
    seed: Optional[int] = None,
    progress: bool = False,
) -> MatchResult:
    """Play ``rounds`` rounds, alternating seats to cancel positional bias."""
    rng = random.Random(seed)
    result = MatchResult()

    for i in range(rounds):
        seat_a = i % 2
        deal_seed = rng.getrandbits(63)
        state = play_round(
            agent_a,
            agent_b,
            rules=rules,
            rng=random.Random(deal_seed),
            seat_a=seat_a,
            first_player=0,
        )
        score_a = state.score(seat_a)
        score_b = state.score(1 - seat_a)
        result.rounds += 1
        result.scores_a.append(score_a)
        result.scores_b.append(score_b)
        result.margins.append(score_b - score_a)
        if score_a < score_b:
            result.wins_a += 1
        elif score_b < score_a:
            result.wins_b += 1
        else:
            result.draws += 1
        if state.knocker == seat_a:
            result.knocks_a += 1
        elif state.knocker is not None:
            result.knocks_b += 1

        if progress and (i + 1) % max(1, rounds // 10) == 0:
            print(f"    ... {i + 1}/{rounds} rounds", flush=True)

    return result
