from dataclasses import dataclass, field
from Skyjo.src.card import Card
from typing import List, Tuple

Pos = Tuple[int, int]


@dataclass
class PlayerState:
    player_id: int
    grid: List[List[Card]] = field(default_factory=list)
    __final_score: int = 0

    def get_grid(self) -> List[List[Card]]:
        return self.grid

    def get_final_game_score(self) -> int:
        return self.__final_score

    def set_final_game_score(self, final_score: int):
        self.__final_score = final_score

    def get_round_score(self) -> int:
        score = 0
        for row in self.grid:
            for card in row:
                if card.face_up:
                    score += card.value
        return score

    def get_hidden_positions(self) -> List[Pos]:
        """
        Get positions of all hidden cards in the player's grid.
        :return: List of tuples representing (row, column) positions of hidden cards
        """
        hidden = []
        for r, row in enumerate(self.grid):
            for c, card in enumerate(row):
                if card and card.is_hidden():
                    hidden.append((r, c))
        return hidden

    def get_all_positions(self) -> List[Pos]:
        """
        Get all positions of cards in the player's grid.
        :return: List of tuples representing (row, column) positions of cards
        """
        pos = []
        for r, row in enumerate(self.grid):
            for c, card in enumerate(row):
                if card is not None:
                    pos.append((r, c))
        return pos

    def reset_round(self, new_grid):
        self.grid = new_grid

    def reset_game(self):
        self.__final_score = 0
        self.grid = []
