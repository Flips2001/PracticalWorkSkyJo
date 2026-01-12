import copy

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
            opponent_scores.append(p.player_state.get_round_score())
        return opponent_scores

    def get_observation(self, player: Player) -> Observation:
        """
        Generates an observation for the given player based on the current game state.
        :param player: Player for whom the observation is generated
        :return: Observation
        """

        return Observation(
            player_id=player.player_id,
            card_grid=copy.deepcopy(player.player_state.get_grid()),
            scores=self.get_players_scores(),
            hand_card=self.game_state.hand_card,
            opponent_cards=self.get_opponent_players_cards(player),
            discard_top=(
                self.game_state.discard_pile[-1]
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
            
            case TurnPhase.HAVE_DRAWN_HIDDEN:
                # If a card is in hand, allow swapping it with any grid position
                for pos in player.player_state.get_all_positions():
                    legal.append(Action(ActionType.SWAP_CARD, pos=pos))
                # Allow discarding the drawn card only if there exists at least one hidden card to flip afterwards
                if player.player_state.get_hidden_positions():
                    legal.append(Action(ActionType.DISCARD_CARD))
                return legal
            
            case TurnPhase.HAVE_DRAWN_OPEN:
                # If a card is in hand, allow swapping it with any grid position
                for pos in player.player_state.get_all_positions():
                    legal.append(Action(ActionType.SWAP_CARD, pos=pos))               
                return legal

            case TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD:
                # Must choose a hidden card to flip
                for pos in player.player_state.get_hidden_positions():
                    legal.append(Action(ActionType.FLIP_CARD, pos=pos))
                return legal

            case TurnPhase.END_TURN:
                return []

        return legal

    def execute_action(self, player: Player, action: Action) -> None:
        """
        Execute the selected action for the given player, mutating game state
        and advancing the turn phase accordingly.
        """
        # Start of turn: choose draw source
        match action.type:
            case ActionType.DRAW_HIDDEN_CARD:
                assert (
                    self.game_state.draw_pile is not None
                ), "Attempted to draw from empty draw pile"
                self.game_state.hand_card = self.game_state.draw_pile.pop()
                self.game_state.hand_card.reveal()
                self.game_state.phase = TurnPhase.HAVE_DRAWN_HIDDEN
                return

            case ActionType.DRAW_OPEN_CARD:
                if self.game_state.phase == TurnPhase.CHOOSE_DRAW:
                    assert (
                        self.game_state.discard_pile is not None
                    ), "Attempted to draw from empty discard pile"
                    self.game_state.hand_card = self.game_state.discard_pile.pop()
                    self.game_state.phase = TurnPhase.HAVE_DRAWN_OPEN
                    return
            # After drawing: either swap the pending card into grid, or discard then flip
            case ActionType.SWAP_CARD:
                if action.pos is None or self.game_state.hand_card is None:
                    raise RuntimeError("Attempted to swap card from empty position")
                r, c = action.pos
                incoming = self.game_state.hand_card
                incoming.reveal()
                outgoing = player.player_state.grid[r][c]
                if outgoing is not None:  # Theoretically should never be None
                    outgoing.reveal()
                    self.game_state.discard_pile.append(outgoing)
                player.player_state.grid[r][c] = incoming
                self.game_state.hand_card = None
                self.game_state.phase = TurnPhase.END_TURN
                return

            case ActionType.DISCARD_CARD:
                hand_card = self.game_state.hand_card
                assert hand_card is not None, "No pending card to discard"
                # Discard the drawn card, then force a flip
                hand_card.reveal()
                self.game_state.discard_pile.append(hand_card)
                self.game_state.hand_card = None
                self.game_state.phase = TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD
                return

            # After discarding, must flip a hidden grid card
            case ActionType.FLIP_CARD:
                assert (
                    action.pos is not None
                ), "Attempted to flip card from empty position"
                r, c = action.pos
                card = player.player_state.grid[r][c]
                if card is not None and card.is_hidden():
                    card.reveal()            
                self.game_state.phase = TurnPhase.END_TURN
                return
        return

    def start_round(self):
        self.game_state.phase = TurnPhase.START_ROUND
        self.game_state.round_start_flips = {i: 0 for i in range(len(self.players))}  # reset

        # Step 1: each player flips 2 cards
        for player in self.players:
            hidden_positions = player.player_state.get_hidden_positions()
            # UI should let player choose 2 positions
            chosen = player.select_starting_flips(hidden_positions, count=2)

            for pos in chosen:
                r, c = pos
                card = player.player_state.grid[r][c]
                card.reveal()

        # Step 2: determine starting player
        self.game_state.current_player_id = self._determine_starting_player()

        # Step 3: transition to first real turn
        self.game_state.phase = TurnPhase.CHOOSE_DRAW
                
    def _determine_starting_player(self) -> int:
        """
        Determine which player starts this round:
        - Primary: highest total score of flipped cards
        - Tie-breaker: highest individual card among flipped cards
        Returns the index of the starting player in self.players
        """
        best_index = 0
        best_score = float("-inf")
        best_highest_card = float("-inf")

        for i, player in enumerate(self.players):
            # Sum of the revealed cards for this round
            score = player.player_state.get_round_score()

            # Highest individual revealed card for tie-break
            highest_card = player.player_state.get_highest_revealed_card()

            print(f"Player {player.player_name} has score {score}, highest card {highest_card}")

            # Choose starting player
            if score > best_score:
                best_score = score
                best_highest_card = highest_card
                best_index = i
            elif score == best_score:
                # Tie-breaker by highest individual card
                if highest_card > best_highest_card:
                    best_highest_card = highest_card
                    best_index = i

        print(f"{self.players[best_index].player_name} will start this round")
        return best_index

    def turn(self, player: Player):
        """
        Plays one full turn for the given player.
        """     
        # Keep asking for actions until the turn is ended by the executed action
        while self.game_state.phase != TurnPhase.END_TURN:
            observation = self.get_observation(player)
            legal_actions = self.get_legal_actions(player)
            selected_action = player.select_action(
                observation=observation, legal_actions=legal_actions
            )
            self.execute_action(player, selected_action)
            self.game_state.remove_unfiorm_columns_to_discard_pile(player.player_state)
        self.game_state.phase = TurnPhase.CHOOSE_DRAW

    def reset(self):
        for player_state in self.get_all_player_states():
            for row in player_state.grid:
                for card in row:
                    card.face_up = True

        # Finish scoring and prepare for next round
        self.game_state.finish_round_and_calculate_stats(self.get_all_player_states())

    def play_game(self):
        while not self.game_state.is_game_over:
            self.play_round()
            self.game_state.game_over()

        print("Game over. Final scores:", self.game_state.all_player_final_scores)

    def play_round(self):
        # Reset the final turn phase at the start of the round
        self.game_state.final_turn_phase = False
        self.game_state.players_to_finish = set()

        self.start_round()
        self.game_state.discard_pile.append(self.game_state.draw_pile.pop())
        self.game_state.discard_pile[-1].reveal()

        round_over = False
        num_players = len(self.players)

        while not round_over:
            current_player = self.players[self.game_state.current_player_id]
            
            # Player takes their turn
            self.turn(current_player)

            # Mark player done if in final turn phase
            if (
                self.game_state.final_turn_phase
                and self.game_state.current_player_id in self.game_state.players_to_finish
            ):
                self.game_state.players_to_finish.remove(self.game_state.current_player_id)

            # Advance to next player
            self.game_state.current_player_id = (self.game_state.current_player_id + 1) % num_players

            # Check if the round is over
            if self.game_state.is_round_over(self.get_all_player_states()):
                round_over = True

        # Reveal all cards at end of round
        self.reset()
