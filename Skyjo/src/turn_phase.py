# Skyjo/src/turn_phase.py
from enum import Enum, auto


class TurnPhase(Enum):
    STARTING_FLIPS = auto() 
    CHOOSE_DRAW = auto()
    HAVE_DRAWN_HIDDEN = auto()
    HAVE_DRAWN_OPEN = auto()
    HAVE_TO_FLIP_AFTER_DISCARD = auto()
    END_TURN = auto()
