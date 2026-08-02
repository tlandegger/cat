"""A 2-player Rat-a-Tat Cat engine and near-optimal solver."""

from .cards import Card, DECK_COMPOSITION, full_deck, parse_card
from .rules import DEFAULT_RULES, RuleConfig
from .state import GameState, Move, MoveKind, Phase, other

__all__ = [
    "Card",
    "DECK_COMPOSITION",
    "full_deck",
    "parse_card",
    "RuleConfig",
    "DEFAULT_RULES",
    "GameState",
    "Move",
    "MoveKind",
    "Phase",
    "other",
]
