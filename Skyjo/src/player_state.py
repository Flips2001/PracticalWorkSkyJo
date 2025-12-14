from dataclasses import dataclass, field
from Skyjo.src.card import Card
from typing import List

@dataclass
class PlayerState:
    player_id: int
    grid: List[List[Card]] = field(default_factory=list)
    __score: int = 0
    __current_score: int = 0
    
    def get_grid(self) -> List[List[Card]]:
        return self.grid

    def get_score(self) -> int:
        return self.__score

    def set_score(self, score: int):
        self.__score = score
    
    def get_current_score(self) -> int:
        return self.__current_score
    
    def calculate_current_score(self) -> int:
        total_score = 0
        for row in self.grid:
            for card in row:
                if card.face_up:
                    total_score += card.value
        self.__current_score = total_score
        return total_score

    def reset(self):
        self.__score = 0
        self.grid = []