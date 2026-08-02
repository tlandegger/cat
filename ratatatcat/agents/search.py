"""Search-based solvers.

Rat-a-Tat Cat cannot be solved exactly. The round has chance nodes on every
deck draw and each player holds private information, so the game tree is both
enormous and partitioned into information sets; exact equilibrium computation
(CFR over the full tree) is not tractable. What *is* tractable, and what
"near-optimal" means in practice for a game of this shape, is sampling.

Two approaches are implemented:

``PIMCAgent`` -- Perfect Information Monte Carlo. Sample a world consistent with
our beliefs, solve that world as if it were a perfect-information game, repeat,
and average each move's value. Simple and strong, but suffers the two textbook
PIMC pathologies: *strategy fusion* (it assumes it can play differently in
worlds it cannot actually tell apart) and *non-locality*. Concretely, it plays
as though it will know the next card off the deck. Useful as a benchmark.

``ISMCTSAgent`` -- Information Set MCTS, the main solver. One tree whose nodes
are information sets rather than states. Each iteration determinizes a fresh
world and walks the *same* tree, so a node accumulates statistics across many
worlds and the agent is forced to commit to a single action per information
set. This removes strategy fusion. Move selection uses UCB with *availability*
counts, which is what makes it correct when different determinizations make
different moves legal -- exactly our situation, since which moves are legal
after a deck draw depends on the card drawn.

This is single-observer ISMCTS (SO-ISMCTS): the tree models the opponent's
decisions from our own viewpoint rather than tracking their information sets
separately. See README for the multi-observer extension.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..cards import Card
from ..state import GameState, Move, MoveKind, Phase, other
from ..belief import determinize, hand_estimates
from .base import Agent
from .heuristic import HeuristicAgent

#: Score margins beyond this are clipped when normalising reward into [0, 1]
#: for UCB. Tuned against the observed margin distribution: heuristic-vs-
#: heuristic rounds have a margin standard deviation of ~7.3 points, so a scale
#: of 12 spreads real outcomes across most of [0, 1]. Setting this too high
#: (25+) squashes the reward spread below the UCB exploration term and the
#: search degenerates towards uniform sampling.
REWARD_SCALE = 12.0


def _normalise(margin: float) -> float:
    """Map a score margin (opponent minus us) onto [0, 1]."""
    return max(0.0, min(1.0, (margin / REWARD_SCALE + 1.0) / 2.0))


def candidate_moves(
    state: GameState, player: int, keep: int = 2
) -> List[Move]:
    """Legal moves with dominated ones removed, to keep the tree narrow.

    The main pruning rule is close to a dominance argument rather than a guess:
    the round is scored as a plain sum, so if you are going to put a given card
    into your hand, putting it over your highest expected-value slot minimises
    the expected total. Lower-value slots are therefore dominated as targets.
    ``keep`` slots are retained rather than one, because replacing an *unknown*
    slot also converts it to known, which carries option value the plain sum
    does not capture.

    Branching at a Swap drops from 16 to ``keep``, which is where most of the
    depth is won.
    """
    legal = state.legal_moves()
    if len(legal) <= 2:
        return legal

    est = hand_estimates(state, player, player)
    worst = sorted(range(state.rules.hand_size), key=lambda s: -est[s])[:keep]

    if state.phase is Phase.TURN_START:
        pruned = [m for m in legal if m.kind is MoveKind.DRAW_DECK]
        pruned += [
            m for m in legal if m.kind is MoveKind.TAKE_DISCARD and m.slot in worst
        ]
        return pruned or legal

    if state.phase is Phase.DECIDE_DRAWN:
        card = state.pending
        assert card is not None
        if card.is_number:
            pruned = [m for m in legal if m.kind is MoveKind.DISCARD_DRAWN]
            pruned += [
                m for m in legal if m.kind is MoveKind.REPLACE and m.slot in worst
            ]
            return pruned or legal
        if card is Card.PEEK:
            unknown = [
                s
                for s in range(state.rules.hand_size)
                if state.knowledge[player][player][s] is None
            ]
            targets = sorted(unknown, key=lambda s: -est[s])[:keep]
            pruned = [m for m in legal if m.kind is MoveKind.DISCARD_DRAWN]
            pruned += [
                m for m in legal if m.kind is MoveKind.USE_PEEK and m.slot in targets
            ]
            return pruned or legal
        if card is Card.SWAP:
            theirs = hand_estimates(state, player, other(player))
            swaps = [m for m in legal if m.kind is MoveKind.USE_SWAP]
            swaps.sort(key=lambda m: theirs[m.opp_slot] - est[m.slot])
            pruned = [m for m in legal if m.kind is MoveKind.DISCARD_DRAWN]
            pruned += swaps[:keep]
            return pruned or legal

    return legal


def _playout(state: GameState, policy: HeuristicAgent, cap: int = 400) -> None:
    """Play ``state`` to the end in place using ``policy`` for both seats."""
    steps = 0
    while not state.is_terminal and steps < cap:
        state.apply(policy.choose(state, state.current))
        steps += 1
    if not state.is_terminal:
        state.phase = Phase.REVEALED


# --------------------------------------------------------------------- ISMCTS
class _Node:
    """One information set. Statistics are stored on the edge's child.

    Deliberately does *not* cache which player is to move. The mover is read
    from the live state during descent instead: when the opponent is played as
    part of the environment, an arbitrary number of their moves happen between
    two of our nodes, so anything recorded at node-creation time is stale by the
    time the node is used for selection -- and getting it wrong flips the sign
    of the value estimate.
    """

    __slots__ = ("children", "visits", "total", "avail")

    def __init__(self) -> None:
        self.children: Dict[Move, "_Node"] = {}
        self.visits: int = 0
        self.total: float = 0.0
        self.avail: int = 0


class ISMCTSAgent(Agent):
    """Information Set Monte Carlo Tree Search."""

    name = "ismcts"

    def __init__(
        self,
        iterations: int = 2000,
        #: Reward spread after normalisation is only ~0.28 standard deviations,
        #: so a textbook 1.4 would drown the signal -- but dropping to 0.25
        #: starves exploration and the search commits to a branch before trying
        #: the alternative. 0.6 measured best of {0.25, 0.6, 1.2}, though the
        #: match differences were inside noise; the decisive argument is the
        #: lock-in case documented in the README.
        exploration: float = 0.6,
        seed: Optional[int] = None,
        use_provenance: bool = True,
        model_opponent: str = "heuristic",
        prune: bool = True,
        prune_keep: int = 2,
        backup: str = "mix",
        pv_weight: float = 0.6,
        pv_min_visit_share: float = 0.06,
    ) -> None:
        self.iterations = iterations
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.use_provenance = use_provenance
        if model_opponent not in ("heuristic", "search"):
            raise ValueError("model_opponent must be 'heuristic' or 'search'")
        #: How the opponent is handled inside the tree.
        #:
        #: ``"heuristic"`` (default) treats their turns as part of the
        #: environment, played by :class:`HeuristicAgent`. ``"search"`` is
        #: textbook SO-ISMCTS, giving them their own UCB nodes.
        #:
        #: Textbook loses here, and measurably gets *worse* with more
        #: iterations: their nodes are visited far too rarely to develop sound
        #: play, so the search converges on a best response to an opponent that
        #: blunders, then meets one that does not. Modelling them with a known
        #: strong policy costs adaptivity and buys back much more in accuracy.
        self.model_opponent = model_opponent
        self.prune = prune
        self.prune_keep = prune_keep
        #: How a root action is scored once the tree is built.
        #:
        #: ``"mean"`` is the textbook average reward. It is systematically
        #: unfair here: calling ends the round almost immediately, so its mean
        #: is a clean estimate, while passing opens a subtree full of our own
        #: future decisions whose mean is dragged down by exploring bad
        #: branches. Measured against ground truth, mean backup estimates
        #: calling to within 0.1 points but undervalues passing by 2.8, biasing
        #: the agent into calling far too often.
        #:
        #: ``"pv"`` scores an action by its best line instead: recurse to the
        #: best-valued child with enough visits to be trustworthy. That removes
        #: the exploration drag but overshoots the other way, because a maximum
        #: over noisy estimates is optimistic -- measured at +3.6 on passing.
        #:
        #: ``"mix"`` (default) blends the two, which is what actually lands near
        #: unbiased; see ``pv_weight``.
        if backup not in ("mean", "pv", "mix"):
            raise ValueError("backup must be 'mean', 'pv' or 'mix'")
        self.backup = backup
        #: Weight on the best-line value under ``backup="mix"``. Calibrated on
        #: 150 knock decisions scored against ground truth: mean backup
        #: undervalues passing by 2.7 points and best-line overvalues it by 3.5.
        #: 0.6 measured best end-to-end (see scripts referenced in the README).
        self.pv_weight = pv_weight
        #: Ignore children with fewer than this share of the parent's visits
        #: when taking the max, so a single lucky rollout cannot define a line.
        self.pv_min_visit_share = pv_min_visit_share
        self.rollout = HeuristicAgent()
        #: Populated after each ``choose`` call: ``{Move: (visits, value)}``,
        #: where value follows ``backup`` and is on the normalised [0, 1] scale.
        #: ``(2 * value - 1) * REWARD_SCALE`` converts it back to score margin.
        self.last_stats: Dict[Move, Tuple[int, float]] = {}

    def choose(self, state: GameState, player: int) -> Move:
        legal = self._moves(state, player)
        if len(legal) == 1:
            self.last_stats = {legal[0]: (0, 0.0)}
            return legal[0]

        root = _Node()
        for _ in range(self.iterations):
            world = determinize(state, player, self.rng, self.use_provenance)
            self._iterate(root, world, player)

        self.last_stats = {
            m: (n.visits, self._value(n)) for m, n in root.children.items()
        }
        if self.backup == "mean":
            # Robust child: most-visited, not highest-mean. A high mean off two
            # visits is noise.
            best = max(root.children.items(), key=lambda kv: (kv[1].visits, kv[1].total))
            return best[0]
        # Principal-variation scoring, with visits breaking near-ties.
        return max(
            root.children.items(),
            key=lambda kv: (round(self._value(kv[1]), 4), kv[1].visits),
        )[0]

    def _value(self, node: _Node) -> float:
        """Score a node by mean reward, best line, or a blend of the two."""
        if not node.visits:
            return 0.0
        mean = node.total / node.visits
        if self.backup == "mean" or not node.children:
            return mean
        floor = node.visits * self.pv_min_visit_share
        trusted = [c for c in node.children.values() if c.visits >= max(floor, 1)]
        if not trusted:
            return mean
        best = max(self._value(c) for c in trusted)
        if self.backup == "pv":
            return best
        return (1.0 - self.pv_weight) * mean + self.pv_weight * best

    def _iterate(self, root: _Node, state: GameState, root_player: int) -> None:
        node = root
        path: List[_Node] = [root]

        # --- select ---------------------------------------------------------
        while not state.is_terminal:
            if self.model_opponent == "heuristic" and state.current != root_player:
                # The opponent is environment, not a searched agent.
                state.apply(self.rollout.choose(state, state.current))
                continue

            legal = self._moves(state, root_player)
            untried = [m for m in legal if m not in node.children]
            if untried:
                move = self.rng.choice(untried)
                child = _Node()
                node.children[move] = child
                # Credit availability to every move we could have taken here.
                for m in legal:
                    if m in node.children:
                        node.children[m].avail += 1
                state.apply(move)
                node = child
                path.append(node)
                break

            for m in legal:
                node.children[m].avail += 1
            # Read the mover from the live state, never from the node.
            mover = state.current
            options = [(m, node.children[m]) for m in legal]
            move, child = max(
                options,
                key=lambda mc: self._selection_value(mc[1], mover, root_player),
            )
            state.apply(move)
            node = child
            path.append(node)

        # --- simulate -------------------------------------------------------
        if not state.is_terminal:
            _playout(state, self.rollout)
        reward = _normalise(state.utility(root_player))

        # --- backpropagate --------------------------------------------------
        for n in path:
            n.visits += 1
            n.total += reward

    def _moves(self, state: GameState, root_player: int) -> List[Move]:
        if self.prune and state.current == root_player:
            return candidate_moves(state, root_player, self.prune_keep)
        return state.legal_moves()

    def _selection_value(self, child: _Node, mover: int, root_player: int) -> float:
        if child.visits == 0:
            return math.inf
        q = child.total / child.visits
        if mover != root_player:
            q = 1.0 - q
        return q + self.exploration * math.sqrt(
            math.log(max(child.avail, 1)) / child.visits
        )


# ----------------------------------------------------------------------- PIMC
class PIMCAgent(Agent):
    """Perfect Information Monte Carlo with depth-limited search per world."""

    name = "pimc"

    def __init__(
        self,
        worlds: int = 120,
        depth: int = 4,
        seed: Optional[int] = None,
        use_provenance: bool = True,
    ) -> None:
        self.worlds = worlds
        self.depth = depth
        self.rng = random.Random(seed)
        self.use_provenance = use_provenance
        self.evaluator = HeuristicAgent()
        self.last_stats: Dict[Move, Tuple[int, float]] = {}

    def choose(self, state: GameState, player: int) -> Move:
        legal = state.legal_moves()
        if len(legal) == 1:
            self.last_stats = {legal[0]: (0, 0.0)}
            return legal[0]

        totals: Dict[Move, float] = {m: 0.0 for m in legal}
        counts: Dict[Move, int] = {m: 0 for m in legal}

        for _ in range(self.worlds):
            world = determinize(state, player, self.rng, self.use_provenance)
            for move in legal:
                branch = world.clone(rng=random.Random(self.rng.random()))
                branch.apply(move)
                totals[move] += self._search(branch, player, self.depth)
                counts[move] += 1

        self.last_stats = {
            m: (counts[m], totals[m] / counts[m] if counts[m] else 0.0) for m in legal
        }
        return max(legal, key=lambda m: totals[m] / max(counts[m], 1))

    def _search(self, state: GameState, root_player: int, depth: int) -> float:
        """Negamax-flavoured search of a determinized (perfect-info) world."""
        if state.is_terminal:
            return float(state.utility(root_player))
        if depth <= 0:
            return self._evaluate(state, root_player)

        legal = state.legal_moves()
        values = []
        for move in legal:
            branch = state.clone(rng=random.Random(self.rng.random()))
            branch.apply(move)
            values.append(self._search(branch, root_player, depth - 1))
        # The mover maximises their own payoff, which in a zero-sum round means
        # maximising ours when it is us and minimising it when it is not.
        return max(values) if state.current == root_player else min(values)

    def _evaluate(self, state: GameState, root_player: int) -> float:
        """Leaf evaluation: finish the world with the heuristic policy."""
        rollout = state.clone(rng=random.Random(self.rng.random()))
        _playout(rollout, self.evaluator)
        return float(rollout.utility(root_player))
