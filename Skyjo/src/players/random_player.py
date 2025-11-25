from players.player import Player
import random

class RandomPlayer(Player):
    def select_action(self, observation, legal_actions):
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")
        return random.choice(legal_actions)