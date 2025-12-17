from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.game_state import GameState
from Skyjo.src.observation import Observation
from Skyjo.src.player_state import PlayerState
from Skyjo.src.players.player import Player

from typing import List, Optional

from Skyjo.src.turn_phase import TurnPhase


class SkyjoGame:
    def __init__(self):
        self.game_state = GameState()
        self.players: List[Player] = []
        self.num_players = 0

    def add_player(self, player: Player):
        player_grid = self.game_state.get_new_player_grid()
        player.player_state.grid = player_grid
        self.players.append(player)
        self.num_players += 1

    def get_all_player_states(self) -> List[PlayerState]:
        return [player.player_state for player in self.players]

    def step(self):
        pass

    def get_opponent_players_cards(
        self, player: Player
    ) -> List[Optional[List[List[Card]]]]:
        """
        Retrieves the cards of all opponent players.
        :param player: The player for whom to get opponent cards
        :return: List of cards for each opponent player
        """
        opponent_cards = [None] * len(self.players)
        for p in self.players:
            if p.player_id != player.player_id:
                opponent_cards[p.player_id] = p.player_state.get_grid()
        return opponent_cards

    def get_players_scores(self) -> List[int]:
        """
        Retrieves the scores of all opponent players.
        :return: List of scores for each opponent player
        """
        opponent_scores = []
        for p in self.players:
            opponent_scores.append(p.player_state.get_score())
        return opponent_scores

    def get_observation(self, player: Player) -> Observation:
        """
        Generates an observation for the given player based on the current game state.
        :param player: Player for whom the observation is generated
        :return: Observation
        """

        return Observation(
            player_id=player.player_id,
            card_grid=player.player_state.get_grid(),
            scores=self.get_players_scores(),
            opponent_cards=self.get_opponent_players_cards(player),
            discard_top=(
                self.game_state.discard_pile[0]
                if len(self.game_state.discard_pile) > 0
                else None
            ),
            draw_pile_size=len(self.game_state.draw_pile),
        )

    def get_legal_actions(self, player: Player) -> List[Action]:
        """
        Determines the legal actions for the given player based on the current game state.
        :param player: Player for whom to determine legal actions
        :return: list of legal actions
        """
        legal: List[Action] = []

        match self.game_state.phase:
            case TurnPhase.CHOOSE_DRAW:
                if self.game_state.draw_pile:
                    legal.append(Action(ActionType.DRAW_HIDDEN_CARD))
                if self.game_state.discard_pile:
                    legal.append(Action(ActionType.DRAW_OPEN_CARD))
                return legal

            case TurnPhase.HAVE_DRAWN:
                # If a card is in hand, allow swapping it with any grid position
                for pos in player.player_state.get_all_positions():
                    legal.append(Action(ActionType.SWAP_CARD, pos=pos))

                # If discarding the drawn card is allowed, represent the follow-up flip.
                for pos in player.player_state.get_all_positions():
                    legal.append(Action(ActionType.DISCARD_AND_FLIP_CARD, pos=pos))
                return legal

        if self.game_state.phase == TurnPhase.END_TURN:
            return []

        return legal

    def turn(self, player: Player):
        """
        Plays one full turn for the given player.
        """
        observation = self.get_observation(player)
        legal_actions = self.get_legal_actions(player)
        selected_action = player.select_action(
            observation=observation, legal_actions=legal_actions
        )
        print(selected_action)
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
            while not self.game_state.is_round_over(self.get_all_player_states()):
                for player in self.players:
                    self.turn(player)
                    if self.game_state.is_round_over(self.get_all_player_states()):
                        break

            self.game_state.calculate_finished_round_stats(self.get_all_player_states())
            self.game_state.game_over()
