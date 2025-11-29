from Skyjo.src.card import Card
from typing import List

class PlayerState:
    score: int
    player_id: int
    grid: List[List[Card]]
    
    def __init__(self, player_id: int):
        self.id = player_id
        self.score = 0
        self.grid: List[List[Card]] = []

    def get_grid(self) -> List[List[Card]]:
        return self.grid
    
    def calculate_score(self) -> int:
        total_score = 0
        for row in self.grid:
            for card in row:
                if card.face_up:
                    total_score += card.value
        self.score = total_score
        return self.score

    def reset(self):
        self.score = 0
        self.grid = []
