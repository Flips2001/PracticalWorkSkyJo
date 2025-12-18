from dataclasses import dataclass
from typing import Optional, Tuple
from Skyjo.src.action_type import ActionType

Pos = Tuple[int, int]


@dataclass(frozen=True)
class Action:
    type: ActionType
    pos: Optional[Pos] = None  # for FLIP_CARD or SWAP target location
    source: Optional[str] = None  # e.g., "draw", "discard" (if needed)

    def __str__(self):
        if self.pos:
            return f"{self.type.__str__()} at pos {self.pos}"
        else:
            return f"{self.type.__str__()}"
