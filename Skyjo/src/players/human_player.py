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

if __name__ == "__main__":
    # Example usage (mocked observation and legal actions)
    from Skyjo.src.card import Card
    from Skyjo.src.observation import Observation

    # Mock observation
    own_grid = [[Card(5, True), Card(3, False)], [Card(2, True), Card(8, False)]]
    visible_opponent_cards = [[[Card(4, True), None], [None, None]], [[None, None], [Card(6, True), None]]]
    opponent_scores = [15, 20]
    discard_top = Card(7, True)
    deck_size = 30

    observation = Observation(
        player_id=0,
        own_grid=own_grid,
        visible_opponent_cards=visible_opponent_cards,
        opponent_scores=opponent_scores,
        discard_top=discard_top,
        deck_size=deck_size
    )

    legal_actions = ["Draw from deck", "Pick up discard"]

    human_player = HumanPlayer(player_id=0, player_name="Alice")
    action = human_player.select_action(observation, legal_actions)
    print(f"Selected action: {action}")