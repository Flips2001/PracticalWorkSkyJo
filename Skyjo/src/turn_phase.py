# Skyjo/src/turn_phase.py
from enum import Enum


class TurnPhase(Enum):
    CHOOSE_DRAW = (
        1  # start of turn: choose from draw pile (hidden) or take open discard
    )
    HAVE_DRAWN = (
        2  # a card is in hand -> swap with a grid card, or discard it and then flip
    )
    HAVE_TO_FLIP_AFTER_DISCARD = (
        3  # after discarding the drawn card, must flip a hidden grid card
    )
    END_TURN = 4  # wrap up (usually auto-advanced)
