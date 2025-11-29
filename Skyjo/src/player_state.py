from dataclasses import dataclass
from Skyjo.src.card import Card
from typing import List

@dataclass
class PlayerState:
    player_id: int
    __score: int = 0
    __grid: List[List[Card]] = []

    def get_grid(self) -> List[List[Card]]:
        return self.__grid
    
    def calculate_score(self) -> int:
        total_score = 0
        for row in self.__grid:
            for card in row:
                if card.face_up:
                    total_score += card.value
        self.score = total_score
        return self.__score

    def reset(self):
        self.__score = 0
        self.__grid = []
