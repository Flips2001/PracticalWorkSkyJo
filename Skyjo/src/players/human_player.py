from players.player import Player

class HumanPlayer(Player):
    def select_action(self, observation, legal_actions):
        if len(legal_actions) == 0:
            raise ValueError("No legal actions available to select from.")

        print(f"Player {self.player_name}, it's your turn!")
        print("Your current grid:")
        for row in observation.own_grid:
            print(" ".join(str(card) for card in row))

        print(f"\nTop of discard pile: {observation.discard_top}")

        print("\nLegal actions:")
        for i, action in enumerate(legal_actions):
            print(f"{i}: {action}")

        while True:
            try:
                choice = int(input("Select the number of your action: "))
                if 0 <= choice < len(legal_actions):
                    return legal_actions[choice]
                else:
                    print("Invalid choice. Please select a valid action number.")
            except ValueError:
                print("Invalid input. Please enter a number.")