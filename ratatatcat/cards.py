"""Card definitions and deck composition for Rat-a-Tat Cat.

Deck (54 cards):
    - Number cards 0..8 : four copies each  (36)
    - Number card 9     : nine copies       ( 9)
    - Peek / Swap / Draw2: three copies each ( 9)
"""

from __future__ import annotations

from collections import Counter
from enum import IntEnum
from typing import Iterable, Iterator, List


class Card(IntEnum):
    """A card is either a number 0-9 or one of three power cards.

    Number cards use their face value as the enum value so that ``int(card)``
    is the score contribution directly. Power cards use sentinel values above
    the number range.
    """

    N0 = 0
    N1 = 1
    N2 = 2
    N3 = 3
    N4 = 4
    N5 = 5
    N6 = 6
    N7 = 7
    N8 = 8
    N9 = 9
    PEEK = 10
    SWAP = 11
    DRAW2 = 12

    @property
    def is_number(self) -> bool:
        # Compare as an int rather than reading ``self.value``: the enum value
        # descriptor is surprisingly costly and this sits in the search hot loop.
        return self < 10

    @property
    def is_power(self) -> bool:
        return self >= 10

    def __str__(self) -> str:  # pragma: no cover - display only
        return _CARD_NAMES[self]


_CARD_NAMES = {
    Card.PEEK: "Peek",
    Card.SWAP: "Swap",
    Card.DRAW2: "Draw2",
    **{Card(v): str(v) for v in range(10)},
}

NUMBER_CARDS: tuple = tuple(Card(v) for v in range(10))
POWER_CARDS: tuple = (Card.PEEK, Card.SWAP, Card.DRAW2)

#: Canonical multiset of the 54-card deck.
DECK_COMPOSITION: Counter = Counter(
    {
        **{Card(v): 4 for v in range(9)},
        Card.N9: 9,
        Card.PEEK: 3,
        Card.SWAP: 3,
        Card.DRAW2: 3,
    }
)

DECK_SIZE = sum(DECK_COMPOSITION.values())  # 54


def full_deck() -> List[Card]:
    """Return a fresh, unshuffled list of all 54 cards."""
    deck: List[Card] = []
    for card, count in DECK_COMPOSITION.items():
        deck.extend([card] * count)
    return deck


def parse_card(text: str) -> Card:
    """Parse a user-typed card token such as ``7``, ``peek``, ``d2``.

    Raises ``ValueError`` on anything unrecognised.
    """
    token = text.strip().lower()
    if token.isdigit():
        value = int(token)
        if 0 <= value <= 9:
            return Card(value)
        raise ValueError(f"number cards run 0-9, got {value!r}")
    aliases = {
        "p": Card.PEEK,
        "peek": Card.PEEK,
        "s": Card.SWAP,
        "swap": Card.SWAP,
        "d": Card.DRAW2,
        "d2": Card.DRAW2,
        "draw2": Card.DRAW2,
        "draw 2": Card.DRAW2,
    }
    if token in aliases:
        return aliases[token]
    raise ValueError(f"unrecognised card {text!r}")


def remaining_counts(seen: Iterable[Card]) -> Counter:
    """Deck composition minus the cards in ``seen``.

    Used by the belief model: every card a player has observed (their own known
    slots, the discard pile, cards revealed by power cards) is subtracted from
    the full deck to give the distribution over each unknown slot.
    """
    counts = Counter(DECK_COMPOSITION)
    counts.subtract(Counter(seen))
    negative = {c: n for c, n in counts.items() if n < 0}
    if negative:
        raise ValueError(f"observed more copies than exist in the deck: {negative}")
    return +counts  # drop zero/negative entries


def iter_cards(counts: Counter) -> Iterator[Card]:
    """Expand a Counter of cards back into individual card instances."""
    for card, n in counts.items():
        for _ in range(n):
            yield card
