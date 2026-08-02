"""Rule configuration.

Published sources for Rat-a-Tat Cat disagree on a handful of edge cases. Every
point where they disagree is a flag here rather than a hard-coded assumption,
so a different table ruling is a one-line change instead of a code hunt.

Defaults are documented in README.md under "Rules and where sources disagree".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleConfig:
    """Tunable rules for a round of Rat-a-Tat Cat."""

    hand_size: int = 4
    #: Slot indices a player is allowed to look at during setup (the outer two).
    initial_peek_slots: tuple = (0, 3)

    # --- Points ------------------------------------------------------------
    #: Score charged for a power card still held at reveal. Only reachable when
    #: ``deal_power_cards`` is True, since power cards otherwise never enter a hand.
    power_card_score: int = 9

    # --- Contested rules ---------------------------------------------------
    #: If False (default) the initial deal is number cards only; power cards
    #: that would have been dealt are shuffled back. This models the published
    #: "a power card in your hand has no power and must be replaced" rule
    #: without the bookkeeping of forced replacement, and keeps every hand slot
    #: in 0..9 -- which is what makes the belief model tractable.
    deal_power_cards: bool = False

    #: If True, a Draw 2 drawn *during* a Draw 2 grants another two draws.
    #: UltraBoardGames says it chains; other summaries say power cards drawn
    #: this way are inert. Default False (inert) -- it always terminates.
    draw2_chains: bool = False

    #: If True, the player who called Rat-a-Tat Cat cannot be targeted by Swap
    #: during the opponent's final turn. Rulebook is silent; default False
    #: (caller is fair game), which matches the "normal play continues" reading.
    knocker_protected: bool = False

    #: A player may not knock before this many of their own turns have completed.
    min_turns_before_knock: int = 1

    #: Hard cap on turns per round, to bound search and catch non-terminating
    #: policies. Reaching it forces an immediate reveal.
    max_turns: int = 200

    def __post_init__(self) -> None:
        if self.hand_size < 2:
            raise ValueError("hand_size must be at least 2")
        for slot in self.initial_peek_slots:
            if not 0 <= slot < self.hand_size:
                raise ValueError(f"initial_peek_slot {slot} outside hand")


DEFAULT_RULES = RuleConfig()
