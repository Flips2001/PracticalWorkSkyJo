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
    assert game_state.all_player_final_scores == []
    assert game_state.final_turn_phase is False
    assert game_state.phase.name == "CHOOSE_DRAW"


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


def test_finish_round_and_calculate_stats(game_state):
    # Create player states with 3x4 grids
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    # Populate each grid with dummy cards
    for ps in [player1, player2]:
        ps.grid = [[Card(0) for _ in range(4)] for _ in range(3)]  # 3x4 grid

    player1.set_final_game_score(50)
    player2.set_final_game_score(70)

    # Also populate the draw pile in game_state with enough cards
    game_state.draw_pile = [Card(i) for i in range(50)]  # plenty of cards

    game_state.finish_round_and_calculate_stats([player1, player2])

    assert game_state.round_number == 2
    assert game_state.all_player_final_scores == [50, 70]
    assert game_state.final_turn_phase is False
    assert getattr(game_state, "first_finisher_id", None) is None

def test_set_final_game_scores(game_state):
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    player1.set_final_game_score(80)
    player2.set_final_game_score(90)

    game_state.all_player_final_scores = game_state.get_all_final_game_scores(
        [player1, player2]
    )
    assert game_state.all_player_final_scores == [80, 90]


def test_get_all_grids(game_state):
    player1 = PlayerState(player_id=1)
    player2 = PlayerState(player_id=2)

    player1.grid = [[Card(1), Card(2)], [Card(3), Card(4)]]
    player2.grid = [[Card(5), Card(6)], [Card(7), Card(8)]]

    grids = game_state.get_all_grids([player1, player2])
    assert grids == [
        [[Card(1), Card(2)], [Card(3), Card(4)]],
        [[Card(5), Card(6)], [Card(7), Card(8)]],
    ]


def test_get_discard_pile(game_state):
    discard_pile = game_state.get_discard_pile()
    assert discard_pile == []  # Initially empty


def test_get_draw_pile(game_state):
    draw_pile = game_state.get_draw_pile()
    assert len(draw_pile) == 150  # Initial deck size

