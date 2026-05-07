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

    # Populate each grid with dummy cards (24 cards total)
    for ps in [player1, player2]:
        ps.grid = [[Card(0) for _ in range(4)] for _ in range(3)]

    player1.set_final_game_score(50)
    player2.set_final_game_score(70)

    # Ensure total cards across draw pile + grids + discard = 150
    # Grids have 24 cards, so draw pile needs 150 - 24 = 126
    game_state.draw_pile = [Card(1) for _ in range(126)]
    game_state.discard_pile = []

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


def test_reset_deck_preserves_all_150_cards(game_state):
    """After reset_deck_from_all_cards, all 150 cards must be in the draw pile."""
    player1 = PlayerState(player_id=0)
    player2 = PlayerState(player_id=1)

    # Simulate mid-game state: cards spread across draw pile, discard, and grids
    # Give each player a 3x4 grid (24 cards total)
    for ps in [player1, player2]:
        ps.grid = [[Card(5) for _ in range(4)] for _ in range(3)]

    # Put some cards in discard pile
    game_state.discard_pile = [Card(3) for _ in range(10)]

    # Remaining cards in draw pile: 150 - 24 - 10 = 116
    game_state.draw_pile = [Card(2) for _ in range(116)]

    game_state.reset_deck_from_all_cards([player1, player2])

    assert len(game_state.draw_pile) == 150
    assert game_state.discard_pile == []
    assert all(not card.face_up for card in game_state.draw_pile)


def test_reset_deck_collects_hand_card(game_state):
    """Hand card should be collected during reset."""
    player1 = PlayerState(player_id=0)
    player2 = PlayerState(player_id=1)

    for ps in [player1, player2]:
        ps.grid = [[Card(0) for _ in range(4)] for _ in range(3)]

    # 1 hand card + 24 grid cards + 125 draw pile = 150
    game_state.hand_card = Card(7)
    game_state.draw_pile = [Card(1) for _ in range(125)]
    game_state.discard_pile = []

    game_state.reset_deck_from_all_cards([player1, player2])

    assert len(game_state.draw_pile) == 150
    assert game_state.hand_card is None


def test_rebuild_draw_pile_from_discard(game_state):
    """When draw pile is empty, it should be rebuilt from discard pile."""
    game_state.draw_pile = []
    # Put 20 cards in discard pile
    game_state.discard_pile = [Card(i % 12) for i in range(20)]
    top_card = game_state.discard_pile[-1]

    game_state._rebuild_draw_pile_from_discard()

    # Draw pile should have 19 cards (all but the top discard card)
    assert len(game_state.draw_pile) == 19
    # Discard pile keeps only the top card
    assert len(game_state.discard_pile) == 1
    assert game_state.discard_pile[0].value == top_card.value
    # All reshuffled cards should be face-down
    assert all(not card.face_up for card in game_state.draw_pile)


def test_draw_card_triggers_reshuffle_when_empty(game_state):
    """Drawing from an empty draw pile should trigger a reshuffle from discard."""
    game_state.draw_pile = []
    game_state.discard_pile = [Card(3), Card(7), Card(11)]

    card = game_state.draw_card()

    # Should have drawn successfully
    assert card is not None
    # Total cards: 1 drawn + remaining draw pile + discard pile top = 3
    assert len(game_state.draw_pile) + len(game_state.discard_pile) + 1 == 3


def test_reset_deck_preserves_draw_pile_cards(game_state):
    """Cards remaining in draw pile at round end must not be lost."""
    player1 = PlayerState(player_id=0)
    player2 = PlayerState(player_id=1)

    for ps in [player1, player2]:
        ps.grid = [[Card(0) for _ in range(4)] for _ in range(3)]

    # Simulate: most cards still in draw pile (typical round end)
    # 24 in grids, 5 in discard, 121 in draw pile = 150
    game_state.draw_pile = [Card(1) for _ in range(121)]
    game_state.discard_pile = [Card(2) for _ in range(5)]

    game_state.reset_deck_from_all_cards([player1, player2])

    assert len(game_state.draw_pile) == 150
