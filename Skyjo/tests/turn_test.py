import pytest

from Skyjo.src.players.random_player import RandomPlayer
from Skyjo.src.skyjo_game import SkyjoGame


@pytest.fixture
def game():
    player = RandomPlayer(player_id=0, player_name="Bot")
    skyjo_game = SkyjoGame()
    skyjo_game.add_player(player)
    return skyjo_game


def test_turn_draw_pile_size(game):
    player = game.players[0]
    initial_draw_pile_size = game.get_observation(player).draw_pile_size
    game.turn(player)
    final_draw_pile_size = game.get_observation(player).draw_pile_size
    # the draw pile size should decrease after the first turn
    assert final_draw_pile_size < initial_draw_pile_size


def test_turn_grid_change(game):
    player = game.players[0]
    initial_card_grid = game.get_observation(player).card_grid
    game.turn(player)
    final_card_grid = game.get_observation(player).card_grid
    # After one turn, the grid should have changed. There has to be one open Card now.
    assert final_card_grid != initial_card_grid


def test_discard_pile_change(game):
    player = game.players[0]
    initial_discard_top = game.get_observation(player).discard_top
    game.turn(player)
    final_discard_top = game.get_observation(player).discard_top
    # After one turn, the top of the discard pile should have changed. There has to be a discard Card now.
    assert final_discard_top != initial_discard_top
