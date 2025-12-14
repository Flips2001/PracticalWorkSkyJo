from Skyjo.src.game_state import GameState
from Skyjo.src.player_state import PlayerState

from typing import List 

class SkyjoGame:
    def __init__(self):
        self.game_state = GameState()
        self.player_states: List[PlayerState] = []   
        self.num_players = 0
    
    def add_player(self, player_state: PlayerState):
        self.player_states.append(player_state)
        self.num_players += 1


    def step(self):
        pass

    def observe(self):
        pass    

    def player_turn(self, player_state: PlayerState):
        pass

    def reset(self):
        self.game_state = GameState()
        for player_state in self.player_states:    
            player_state.reset()  

    def play_round(self):
        self.reset()
        while not self.game_state.is_game_over:
            while not self.game_state.is_round_over(self.player_states):
                for player_state in self.player_states:
                    self.player_turn(player_state)    
                    if self.game_state.is_round_over(self.player_states):
                        break
                    
            self.game_state.calculate_finished_round_stats(self.player_states)
            self.game_state.game_over()

 

