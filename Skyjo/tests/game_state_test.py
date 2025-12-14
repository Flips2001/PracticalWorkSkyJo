import pytest
from Skyjo.src.card import Card
from Skyjo.src.player_state import PlayerState

@pytest.fixture
def game_state():
    from Skyjo.src.game_state import GameState
    return GameState()

def test_initial_game_state(game_state):
    assert game_state.current_player_id == 0
    assert game_state.round_number == 1
    assert game_state.is_game_over is False
    assert game_state.discard_pile == []
    assert game_state.draw_pile != []  # Deck should be created on init

def test_create_deck(game_state):
    deck = game_state.create_deck()
    assert len(deck) == 150  # 10 x (-1 to 12) + 5 x -2 + 15 x 0
    value_counts = {value: 0 for value in range(-2, 13)}
    for card in deck:
        value_counts[card.value] += 1
    
    for value in range(1, 13):
        assert value_counts[value] == 10
    assert value_counts[-2] == 5
    assert value_counts[0] == 15
    assert value_counts[-1] == 10
    
def test_is_round_over(game_state):

    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    # Set up grids with all cards face-up
    player1.grid = [[Card(5) for _ in range(3)] for _ in range(3)]
    player2.grid = [[Card(10) for _ in range(3)] for _ in range(3)]
    for row in player1.grid:
        for card in row:
            card.reveal()
    for row in player2.grid:
        for card in row:
            card.reveal()

    assert game_state.is_round_over([player1, player2]) is True

    # Now hide one card
    player1.grid[0][0] = Card(5)  # New card, face-down by default
    assert game_state.is_round_over([player1, player2]) is False

def test_calculate_finished_round_stats(game_state):
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    player1.set_score(30)
    player2.set_score(45)

    game_state.calculate_finished_round_stats([player1, player2])

    assert game_state.all_player_scores == [30, 45]
    assert game_state.round_number == 2

def test_get_all_scores(game_state):
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    player1.set_score(20)
    player2.set_score(35)

    scores = game_state.get_all_scores([player1, player2])
    assert scores == [20, 35]

def test_get_all_grids(game_state):
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    player1.grid = [[Card(1), Card(2)], [Card(3), Card(4)]]
    player2.grid = [[Card(5), Card(6)], [Card(7), Card(8)]]

    grids = game_state.get_all_grids([player1, player2])
    assert grids == [
        [[Card(1), Card(2)], [Card(3), Card(4)]],
        [[Card(5), Card(6)], [Card(7), Card(8)]]
    ]

def test_game_over(game_state):
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    player1.set_score(95)
    player2.set_score(85)

    game_state.all_player_scores = game_state.get_all_scores([player1, player2])
    game_state.game_over()
    assert game_state.is_game_over is False

    player1.set_score(105)
    game_state.all_player_scores = game_state.get_all_scores([player1, player2])
    game_state.game_over()
    assert game_state.is_game_over is True

def test_get_discard_pile(game_state):
    discard_pile = game_state.get_discard_pile()
    assert discard_pile == []  # Initially empty

def test_get_draw_pile(game_state):
    draw_pile = game_state.get_draw_pile()
    assert len(draw_pile) == 150  # Initial deck size

def test_round_number_increment(game_state):
    initial_round = game_state.round_number
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)
    game_state.calculate_finished_round_stats([player1, player2])
    assert game_state.round_number == initial_round + 1