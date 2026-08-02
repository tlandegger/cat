"""Game state, move generation and transitions for 2-player Rat-a-Tat Cat.

The state holds ground truth (every card face-up to the engine) *plus* a
per-player knowledge table recording what each player legitimately knows. The
knowledge table is what makes correct determinization possible: a search agent
never reads the ground-truth hands, only ``knowledge[me]``.

Turn flow, as a phase machine::

    TURN_START ──take discard(slot)──────────────► KNOCK_DECISION
               └─draw deck─► [chance] ─► DECIDE_DRAWN ─┬─replace(slot)─► KNOCK_DECISION
                                                       ├─discard───────► KNOCK_DECISION
                                                       ├─use peek(slot)► KNOCK_DECISION
                                                       ├─use swap(i,j)─► KNOCK_DECISION
                                                       └─use draw2 ────► [chance] ─► DECIDE_DRAWN
                                                                          (draws_left decremented)

    KNOCK_DECISION ──knock / pass──► next player's TURN_START (or reveal)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

from .cards import Card, full_deck
from .rules import DEFAULT_RULES, RuleConfig

NUM_PLAYERS = 2


class Phase(Enum):
    TURN_START = auto()
    DECIDE_DRAWN = auto()
    KNOCK_DECISION = auto()
    REVEALED = auto()


class MoveKind(Enum):
    TAKE_DISCARD = auto()
    DRAW_DECK = auto()
    REPLACE = auto()
    DISCARD_DRAWN = auto()
    USE_PEEK = auto()
    USE_SWAP = auto()
    USE_DRAW2 = auto()
    KNOCK = auto()
    PASS_KNOCK = auto()


@dataclass(frozen=True)
class Move:
    kind: MoveKind
    slot: int = -1
    opp_slot: int = -1

    def __str__(self) -> str:  # pragma: no cover - display only
        k = self.kind
        if k is MoveKind.TAKE_DISCARD:
            return f"take discard -> slot {self.slot}"
        if k is MoveKind.DRAW_DECK:
            return "draw from deck"
        if k is MoveKind.REPLACE:
            return f"keep drawn card -> slot {self.slot}"
        if k is MoveKind.DISCARD_DRAWN:
            return "discard drawn card"
        if k is MoveKind.USE_PEEK:
            return f"use Peek on own slot {self.slot}"
        if k is MoveKind.USE_SWAP:
            return f"use Swap: my slot {self.slot} <-> their slot {self.opp_slot}"
        if k is MoveKind.USE_DRAW2:
            return "use Draw 2"
        if k is MoveKind.KNOCK:
            return 'call "Rat-a-Tat Cat!"'
        return "do not call"


def other(player: int) -> int:
    return 1 - player


@dataclass
class GameState:
    """Full ground-truth state of one round."""

    rules: RuleConfig = DEFAULT_RULES
    hands: List[List[Card]] = field(default_factory=list)
    deck: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)

    #: knowledge[observer][owner][slot] -> Card the observer knows sits there,
    #: or None if unknown to them.
    knowledge: List[List[List[Optional[Card]]]] = field(default_factory=list)

    #: provenance[owner][slot] -> how the card currently there arrived. This is
    #: public: at a real table everyone sees which slot was replaced, peeked or
    #: swapped, even when the card's face stays hidden. The belief model uses it
    #: to infer value (a card its owner chose to keep is probably low).
    provenance: List[List[str]] = field(default_factory=list)

    current: int = 0
    phase: Phase = Phase.TURN_START
    pending: Optional[Card] = None
    #: Extra draws still available in the current Draw 2 sequence.
    draws_left: int = 0
    #: True for the whole duration of a Draw 2 sequence. Distinct from
    #: ``draws_left`` because a power card drawn on the *last* sub-draw is still
    #: inert, at which point ``draws_left`` is already 0.
    in_draw2: bool = False
    knocker: Optional[int] = None
    turns_taken: List[int] = field(default_factory=lambda: [0, 0])
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    # ------------------------------------------------------------------ setup
    @classmethod
    def deal(
        cls,
        rules: RuleConfig = DEFAULT_RULES,
        rng: Optional[random.Random] = None,
        first_player: int = 0,
    ) -> "GameState":
        rng = rng or random.Random()
        deck = full_deck()
        rng.shuffle(deck)

        hands: List[List[Card]] = []
        if rules.deal_power_cards:
            for _ in range(NUM_PLAYERS):
                hands.append([deck.pop() for _ in range(rules.hand_size)])
        else:
            # Deal number cards only; power cards passed over go back and are
            # reshuffled once both hands are complete.
            skipped: List[Card] = []
            for _ in range(NUM_PLAYERS):
                hand: List[Card] = []
                while len(hand) < rules.hand_size:
                    card = deck.pop()
                    (hand if card.is_number else skipped).append(card)
                hands.append(hand)
            deck.extend(skipped)
            rng.shuffle(deck)

        discard = [deck.pop()]

        knowledge: List[List[List[Optional[Card]]]] = [
            [[None] * rules.hand_size for _ in range(NUM_PLAYERS)]
            for _ in range(NUM_PLAYERS)
        ]
        provenance = [["dealt"] * rules.hand_size for _ in range(NUM_PLAYERS)]
        for p in range(NUM_PLAYERS):
            for slot in rules.initial_peek_slots:
                knowledge[p][p][slot] = hands[p][slot]
                provenance[p][slot] = "dealt_seen"

        return cls(
            rules=rules,
            hands=hands,
            deck=deck,
            discard=discard,
            knowledge=knowledge,
            provenance=provenance,
            current=first_player,
            rng=rng,
        )

    # ------------------------------------------------------------------ clone
    def clone(self, rng: Optional[random.Random] = None) -> "GameState":
        return GameState(
            rules=self.rules,
            hands=[h[:] for h in self.hands],
            deck=self.deck[:],
            discard=self.discard[:],
            knowledge=[[slots[:] for slots in obs] for obs in self.knowledge],
            provenance=[p[:] for p in self.provenance],
            current=self.current,
            phase=self.phase,
            pending=self.pending,
            draws_left=self.draws_left,
            in_draw2=self.in_draw2,
            knocker=self.knocker,
            turns_taken=self.turns_taken[:],
            rng=rng if rng is not None else random.Random(self.rng.random()),
        )

    # ------------------------------------------------------------- properties
    @property
    def is_terminal(self) -> bool:
        return self.phase is Phase.REVEALED

    @property
    def discard_top(self) -> Optional[Card]:
        return self.discard[-1] if self.discard else None

    def score(self, player: int) -> int:
        return sum(
            int(c) if c.is_number else self.rules.power_card_score
            for c in self.hands[player]
        )

    def utility(self, player: int) -> int:
        """Zero-sum payoff: opponent's score minus ours. Higher is better.

        Using the margin rather than win/loss matches the real game, which is
        scored cumulatively across rounds.
        """
        return self.score(other(player)) - self.score(player)

    # ---------------------------------------------------------------- moves
    def legal_moves(self) -> List[Move]:
        if self.phase is Phase.REVEALED:
            return []
        hand_size = self.rules.hand_size

        if self.phase is Phase.TURN_START:
            moves: List[Move] = [Move(MoveKind.DRAW_DECK)]
            top = self.discard_top
            # Only number cards may be taken from the discard pile; a discarded
            # power card is spent and out of play.
            if top is not None and top.is_number:
                moves.extend(Move(MoveKind.TAKE_DISCARD, s) for s in range(hand_size))
            return moves

        if self.phase is Phase.DECIDE_DRAWN:
            assert self.pending is not None
            card = self.pending
            if card.is_number:
                return [Move(MoveKind.DISCARD_DRAWN)] + [
                    Move(MoveKind.REPLACE, s) for s in range(hand_size)
                ]
            # A power card drawn mid-Draw-2 is inert unless chaining is enabled.
            inert = self.in_draw2 and not self.rules.draw2_chains
            if inert:
                return [Move(MoveKind.DISCARD_DRAWN)]
            if card is Card.PEEK:
                return [Move(MoveKind.DISCARD_DRAWN)] + [
                    Move(MoveKind.USE_PEEK, s) for s in range(hand_size)
                ]
            if card is Card.SWAP:
                victim = other(self.current)
                if self.rules.knocker_protected and self.knocker == victim:
                    return [Move(MoveKind.DISCARD_DRAWN)]
                return [Move(MoveKind.DISCARD_DRAWN)] + [
                    Move(MoveKind.USE_SWAP, i, j)
                    for i in range(hand_size)
                    for j in range(hand_size)
                ]
            return [Move(MoveKind.DISCARD_DRAWN), Move(MoveKind.USE_DRAW2)]

        # KNOCK_DECISION
        moves = [Move(MoveKind.PASS_KNOCK)]
        if self.knocker is None and self.turns_taken[self.current] >= self.rules.min_turns_before_knock:
            moves.append(Move(MoveKind.KNOCK))
        return moves

    # ------------------------------------------------------------ transitions
    def apply(self, move: Move) -> None:
        """Mutate the state by playing ``move``. Chance outcomes use ``self.rng``."""
        if self.phase is Phase.REVEALED:
            raise RuntimeError("round is over")
        k = move.kind
        me = self.current

        if k is MoveKind.DRAW_DECK:
            self._draw_into_pending()
            return

        if k is MoveKind.TAKE_DISCARD:
            taken = self.discard.pop()
            replaced = self.hands[me][move.slot]
            self.hands[me][move.slot] = taken
            self.discard.append(replaced)
            # Taking from the discard is fully public: both players see exactly
            # which card landed in that slot.
            for observer in range(NUM_PLAYERS):
                self.knowledge[observer][me][move.slot] = taken
            self.provenance[me][move.slot] = "took_discard"
            self._end_turn()
            return

        if k is MoveKind.REPLACE:
            assert self.pending is not None
            replaced = self.hands[me][move.slot]
            self.hands[me][move.slot] = self.pending
            self.discard.append(replaced)
            self.knowledge[me][me][move.slot] = self.pending
            # The opponent saw the outgoing card but not the incoming one.
            self.knowledge[other(me)][me][move.slot] = None
            self.provenance[me][move.slot] = "kept_from_deck"
            self.pending = None
            # Keeping a card forfeits any remaining Draw 2 sub-draws.
            self._end_turn()
            return

        if k is MoveKind.DISCARD_DRAWN:
            assert self.pending is not None
            self.discard.append(self.pending)
            self.pending = None
            self._continue_or_end()
            return

        if k is MoveKind.USE_PEEK:
            self.discard.append(self.pending)  # type: ignore[arg-type]
            self.pending = None
            self.knowledge[me][me][move.slot] = self.hands[me][move.slot]
            if self.provenance[me][move.slot] == "dealt":
                self.provenance[me][move.slot] = "dealt_seen"
            self._continue_or_end()
            return

        if k is MoveKind.USE_SWAP:
            victim = other(me)
            i, j = move.slot, move.opp_slot
            self.discard.append(self.pending)  # type: ignore[arg-type]
            self.pending = None
            self.hands[me][i], self.hands[victim][j] = (
                self.hands[victim][j],
                self.hands[me][i],
            )
            # Neither player sees the swapped cards, so each observer's beliefs
            # simply travel with the cards.
            for observer in range(NUM_PLAYERS):
                kn = self.knowledge[observer]
                kn[me][i], kn[victim][j] = kn[victim][j], kn[me][i]
            # What a card's history implies about its value travels with it.
            self.provenance[me][i], self.provenance[victim][j] = (
                self.provenance[victim][j],
                self.provenance[me][i],
            )
            self._continue_or_end()
            return

        if k is MoveKind.USE_DRAW2:
            self.discard.append(self.pending)  # type: ignore[arg-type]
            self.pending = None
            self.in_draw2 = True
            # Two draws total: the one taken immediately, plus one held back.
            self.draws_left += 1
            self._draw_into_pending()
            return

        if k is MoveKind.KNOCK:
            self.knocker = me
            self._advance_player()
            return

        if k is MoveKind.PASS_KNOCK:
            self._advance_player()
            return

        raise ValueError(f"unhandled move {move!r}")

    # ------------------------------------------------------------- internals
    def _draw_into_pending(self) -> None:
        """Draw a card into ``pending``, or reveal hands if nothing is left."""
        if not self.deck:
            self._reshuffle()
        if not self.deck:
            # Unreachable with a legal 54-card deck (46 cards always circulate),
            # but ending cleanly beats raising out of the middle of a search.
            self.phase = Phase.REVEALED
            return
        self.pending = self.deck.pop()
        self.phase = Phase.DECIDE_DRAWN

    def _continue_or_end(self) -> None:
        """Take the next Draw-2 sub-draw if one is owed, otherwise end the turn."""
        if self.draws_left > 0:
            self.draws_left -= 1
            self._draw_into_pending()
            return
        self._end_turn()

    def _reshuffle(self) -> None:
        """Turn the spent discard pile back into a draw pile, keeping its top."""
        if len(self.discard) <= 1:
            return
        top = self.discard.pop()
        self.deck = self.discard
        self.discard = [top]
        self.rng.shuffle(self.deck)

    def _end_turn(self) -> None:
        self.draws_left = 0
        self.in_draw2 = False
        self.pending = None
        self.turns_taken[self.current] += 1
        self.phase = Phase.KNOCK_DECISION

    def _advance_player(self) -> None:
        self.current = other(self.current)
        if self.knocker is not None and self.current == self.knocker:
            self.phase = Phase.REVEALED
            return
        if sum(self.turns_taken) >= self.rules.max_turns:
            self.phase = Phase.REVEALED
            return
        self.phase = Phase.TURN_START

    # ---------------------------------------------------------------- display
    def render(self, viewpoint: Optional[int] = None) -> str:  # pragma: no cover
        lines = []
        for p in range(NUM_PLAYERS):
            if viewpoint is None:
                cells = [str(c) for c in self.hands[p]]
            else:
                cells = [
                    str(c) if (c := self.knowledge[viewpoint][p][s]) is not None else "?"
                    for s in range(self.rules.hand_size)
                ]
            tag = "you" if p == viewpoint else f"P{p}"
            lines.append(f"  {tag:>4}: [{' '.join(f'{c:>2}' for c in cells)}]")
        lines.append(f"  discard top: {self.discard_top}   deck: {len(self.deck)}")
        return "\n".join(lines)
