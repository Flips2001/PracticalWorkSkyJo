from Skyjo.src.players.player import Player
import random


class RandomPlayer(Player):
    def select_action(self, observation, legal_actions):
        """
        Randomly selects an action from the list of legal actions.
        Args:
            observation: An object containing the current state of the game relevant to the player.
            legal_actions: A list of actions that the player is allowed to take on this turn.
        Returns:
            The action randomly selected from the list of legal actions.
        Raises:
            ValueError: If there are no legal actions available to select from.
        """

        if not legal_actions:
            raise ValueError("No legal actions available to select from.")
        return random.choice(legal_actions)
