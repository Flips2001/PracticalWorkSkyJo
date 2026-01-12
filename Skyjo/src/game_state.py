import random
from dataclasses import dataclass
from typing import List, Optional
from Skyjo.src.card import Card
from Skyjo.src.player_state import PlayerState
from Skyjo.src.turn_phase import TurnPhase


@dataclass
class GameState:
    round_number: int
    discard_pile: List[Card]
    draw_pile: List[Card]
    current_player_id: int
    is_game_over: bool
    all_player_final_scores: List[int]
    final_turn_phase: bool
    phase: TurnPhase
    hand_card: Optional[Card]    
    round_start_flips: dict[int, int] 

    def __init__(self):
        self.round_number = 1
        self.discard_pile = []
        self.draw_pile = self.create_deck()
        self.current_player_id = 0
        self.is_game_over = False
        self.all_player_final_scores = []
        self.final_turn_phase = False
        self.phase = TurnPhase.CHOOSE_DRAW
        self.hand_card = None
        self.round_start_flips: dict[int, int] = {}  # player_id -> flips done this round

    def create_deck(self) -> List[Card]:
        deck: List[Card] = []
        # cards -1 to 12 (10 of each)
        for _ in range(10):
            for value in range(-1, 13):
                deck.append(Card(value))
        # 5 cards with value -2
        for _ in range(5):
            deck.append(Card(-2))
        # 5 more cards with value 0 (15 in total)
        for _ in range(5):
            deck.append(Card(0))
        random.shuffle(deck)
        return deck

    def get_discard_pile(self) -> List[Optional[Card]]:
        return self.discard_pile

    def get_draw_pile(self) -> List[Card]:
        return self.draw_pile

    def get_all_final_game_scores(self, player_states: List[PlayerState]) -> List[int]:
        return [player_state.get_final_game_score() for player_state in player_states]

    def get_all_grids(self, player_states: List[PlayerState]) -> List[List[List[Card]]]:
        return [player_state.grid for player_state in player_states]
    
    def is_column_uniform(self, player_state: PlayerState, col: int) -> bool:
        """
        Check if all cards in a given column are the same value and face-up.
        :param player_state: The PlayerState containing the grid to check.
        :param col: The column index to check (0-3).
        :return: True if all cards in the column are the same value and face-up, False otherwise.
        """
        grid = player_state.get_grid()
        first_card = grid[0][col]
        if not first_card.face_up:
            return False
        for row in range(1, len(grid)):
            card = grid[row][col]
            if not card.face_up or card.value != first_card.value:
                return False
        return True

    def remove_unfiorm_columns_to_discard_pile(self, player_state: PlayerState):
        """
        Remove columns from the player's grid where all cards are the same value and face-up.
        :param player_state: The PlayerState containing the grid to modify.
        """
        grid = player_state.get_grid()
        if not grid or not grid[0]:
            return
        
        num_cols = len(grid[0])
        num_rows = len(grid)

        cols_to_remove = [
            col for col in range(num_cols)
            if self.is_column_uniform(player_state, col)
        ]
        for col in reversed(cols_to_remove):
            for row in range(num_rows):
                self.discard_pile.append(grid[row].pop(col))
          
    def get_new_player_grid(self) -> List[List[Card]]:
        """
        Generates a new 3x4 grid of cards for a player by drawing from the draw pile.
        :return: A 3x4 grid (list of lists) of Card objects.
        """
        grid: List[List[Card]] = []
        for _ in range(3):
            row: List[Card] = []
            for _ in range(4):
                row.append(self.draw_pile.pop())
            grid.append(row)
        return grid

    def is_round_over(self, player_states: List[PlayerState]) -> bool:
        # Check if any player has all cards face-up
        for i, player_state in enumerate(player_states):
            grid = player_state.grid
            if all(card.face_up for row in grid for card in row):
                # Start final turn phase if not already started
                if not self.final_turn_phase:
                    self.final_turn_phase = True
                    # All other players get one last move
                    self.players_to_finish = set(
                        j for j in range(len(player_states)) if j != i
                    )
                break  # first player to finish found

        # If in final turn phase, round ends only when all remaining players finished
        if self.final_turn_phase:
            return len(self.players_to_finish) == 0

        # Round not over yet
        return False

    def finish_round_and_calculate_stats(self, player_states: List[PlayerState]):

        for player_state in player_states:
            player_state.set_final_game_score(
                player_state.get_final_game_score() + player_state.get_round_score()
            )

        self.all_player_final_scores = self.get_all_final_game_scores(player_states)
        print(
            f"\nEnd of Round {self.round_number} Scores: {self.all_player_final_scores}\n"
        )

        for player_state in player_states:
            new_grid = self.get_new_player_grid()
            player_state.reset_round(new_grid)
        self.round_number += 1
        self.draw_pile = self.create_deck()
        self.discard_pile = []

    def game_over(self):
        if any(score >= 100 for score in self.all_player_final_scores):
            self.is_game_over = True
