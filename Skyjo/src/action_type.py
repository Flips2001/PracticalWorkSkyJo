from enum import Enum


class ActionType(Enum):
    DRAW_HIDDEN_CARD = 1
    DRAW_OPEN_CARD = 2
    FLIP_CARD = 3
    SWAP_CARD = 4
    DISCARD_CARD = 5

    def __str__(self):
        return self.name.lower().replace("_", " ")
