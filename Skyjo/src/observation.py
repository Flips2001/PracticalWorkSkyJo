from dataclasses import dataclass
from typing import Optional

from Skyjo.src.card import Card
from Skyjo.src.turn_phase import TurnPhase


@dataclass(frozen=True, slots=True)
class ObservedCard:
    """Immutable, player-visible representation of a card."""

    value: Optional[int]
    face_up: bool

    @classmethod
    def from_card(cls, card: Card) -> "ObservedCard":
        if card.face_up:
            return cls(value=card.get_value(), face_up=True)
        return cls(value=None, face_up=False)

    def is_hidden(self) -> bool:
        return not self.face_up

    def get_value(self) -> int:
        if self.value is not None:
            return self.value
        raise ValueError("Card is face down; value is not accessible.")

    def __repr__(self) -> str:
        if self.face_up:
            return f"[{self.get_value()}]"
        return "[X]"


ObservedGrid = tuple[tuple[ObservedCard, ...], ...]


def _freeze_grid(grid) -> ObservedGrid:
    return tuple(
        tuple(
            card if isinstance(card, ObservedCard) else ObservedCard.from_card(card)
            for card in row
        )
        for row in grid
    )


def _freeze_card(card) -> Optional[ObservedCard]:
    if card is None or isinstance(card, ObservedCard):
        return card
    return ObservedCard.from_card(card)


@dataclass(frozen=True, slots=True)
class Observation:
    player_id: int
    card_grid: ObservedGrid
    hand_card: Optional[ObservedCard]
    opponent_cards: tuple[Optional[ObservedGrid], ...]
    scores: tuple[int, ...]
    discard_top: Optional[ObservedCard]
    draw_pile_size: int
    turn_phase: TurnPhase
    discard_pile_value_counts: Optional[tuple[int, ...]] = None
    total_scores: tuple[int, ...] = ()
    final_turn_phase: bool = False
    first_finisher_id: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "card_grid", _freeze_grid(self.card_grid))
        object.__setattr__(self, "hand_card", _freeze_card(self.hand_card))
        object.__setattr__(self, "discard_top", _freeze_card(self.discard_top))
        object.__setattr__(
            self,
            "opponent_cards",
            tuple(
                None if grid is None else _freeze_grid(grid)
                for grid in self.opponent_cards
            ),
        )
        object.__setattr__(self, "scores", tuple(self.scores))
        object.__setattr__(self, "total_scores", tuple(self.total_scores or ()))
        if self.discard_pile_value_counts is not None:
            object.__setattr__(
                self,
                "discard_pile_value_counts",
                tuple(self.discard_pile_value_counts),
            )
