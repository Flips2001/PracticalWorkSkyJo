from Skyjo.src.observation import Observation
from Skyjo.src.players.player import Player


class HumanPlayer(Player):
    def select_starting_flips(self, hidden_positions, count=2):
        print(f"\n{self.player_name}, select {count} cards to flip (from hidden positions):")
        for i, pos in enumerate(hidden_positions):
            print(f"{i}: {pos}")

        chosen_indices = []
        while len(chosen_indices) < count:
            try:
                choice = int(input(f"Pick card {len(chosen_indices)+1}: "))
                if choice in chosen_indices or choice < 0 or choice >= len(hidden_positions):
                    print("Invalid or duplicate choice.")
                    continue
                chosen_indices.append(choice)
            except ValueError:
                print("Enter a valid number.")

        return [hidden_positions[i] for i in chosen_indices]
    def select_action(self, observation: Observation, legal_actions):
        """
        Prompt the human player to select an action from the list of legal actions.
        Displays the player's current grid, the top of the discard pile, and the available legal actions.
        Prompts the user to input the number corresponding to their chosen action.
        Args:
            observation: An object containing the current state of the game relevant to the player.
            legal_actions: A list of actions that the player is allowed to take on this turn.
        Returns:
            The action selected by the player from the list of legal actions.
        Raises:
            ValueError: If there are no legal actions available to select from.
        """

        if not legal_actions:
            raise ValueError("No legal actions available to select from.")

        print(f"Player {self.player_name}, it's your turn!")
        print("Your current grid:")
        for row in observation.card_grid:
            print(" ".join(str(card) for card in row))

        print(f"\nYour hand card: {observation.hand_card}")
        print(f"Top of discard pile: {observation.discard_top}")

        print(f"\nYour score: {observation.scores[observation.player_id]}")

        print("\nLegal actions:")
        for i, action in enumerate(legal_actions):
            print(f"{i}: {action}")

        while True:
            try:
                choice = int(input("\nSelect the number of your action: "))
                if 0 <= choice < len(legal_actions):
                    return legal_actions[choice]
                else:
                    print("Invalid choice. Please select a valid action number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
