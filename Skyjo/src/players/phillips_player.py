import random

from Skyjo.src.action_type import ActionType
from Skyjo.src.players.player import Player
from Skyjo.src.turn_phase import TurnPhase


class PhillipsPlayer(Player):
    """
    A Skyjo player that follows a simple strategy:

    - Chooses to draw from the discard pile if the top card has a value of 5 or less; otherwise, draws from the draw pile.

    - After drawing, prefers to swap the hand card with the highest value card in the grid if it improves the grid.

    - If the hand card is greater than 5, discard it and flip a random card; otherwise, exchange it with a hidden card in the grid.

    - If no other strategy applies, select the first legal action available.
    """

    def select_action(self, observation, legal_actions):

        if observation.turn_phase == TurnPhase.CHOOSE_DRAW:
            # Choose to draw from discard pile if the top card has value 5 or less
            if (
                observation.discard_top is not None
                and observation.discard_top.value <= 5
            ):
                for action in legal_actions:
                    if action.type == ActionType.DRAW_OPEN_CARD:
                        return action

            # Otherwise, choose to draw from draw pile
            else:
                for action in legal_actions:
                    if action.type == ActionType.DRAW_HIDDEN_CARD:
                        return action

        if (
            observation.turn_phase == TurnPhase.HAVE_DRAWN_HIDDEN
            or observation.turn_phase == TurnPhase.HAVE_DRAWN_OPEN
        ):
            highest_card_value, pos = max(
                (
                    (card.value, (i, j))
                    for i, row in enumerate(observation.card_grid)
                    for j, card in enumerate(row)
                ),
                key=lambda x: x[0],
            )

            # Prefer to play the hand card if it improves the grid
            if (
                observation.hand_card is not None
                and observation.hand_card.value < highest_card_value
            ):
                for action in legal_actions:
                    if action.type == ActionType.SWAP_CARD and action.pos == pos:
                        return action
            # If the hand card is greater than 5, discard it and flip a random card
            elif observation.hand_card is not None and observation.hand_card.value > 5:
                for action in legal_actions:
                    if action.type == ActionType.DISCARD_CARD:
                        return action
            # Otherwise, exchange it with a hidden card in the grid
            else:
                pos = next(
                    (
                        (i, j)
                        for i, row in enumerate(observation.card_grid)
                        for j, card in enumerate(row)
                        if card.is_hidden()
                    ),
                    None,
                )
                for action in legal_actions:
                    if action.type == ActionType.SWAP_CARD and action.pos == pos:
                        return action

        # Always select the first legal action
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")
        return random.choice(legal_actions)
