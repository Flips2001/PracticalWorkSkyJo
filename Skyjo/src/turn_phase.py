# Skyjo/src/turn_phase.py
from enum import Enum


class TurnPhase(Enum):
    CHOOSE_DRAW = (
        1  # start of turn: choose from draw pile (hidden) or take open discard
    )
    HAVE_DRAWN = 2  # you have a card in hand -> either swap with a grid card, or discard it and flip
    END_TURN = 3  # wrap up (usually auto-advanced)
