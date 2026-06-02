from pathlib import Path

import pytest

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.players.player import Player
from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.turn_phase import TurnPhase


CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "rl"
    / "checkpoints"
    / "skyjo_ppo_final.zip"
)


class PassivePlayer(Player):
    def select_action(self, observation, legal_actions):
        return legal_actions[0]


def _revealed_card(value: int) -> Card:
    card = Card(value)
    card.reveal()
    return card


def _revealed_grid(values: list[list[int]]) -> list[list[Card]]:
    return [[_revealed_card(value) for value in row] for row in values]


def _make_column_clear_game(
    model, run_index: int
) -> tuple[SkyjoGame, Player, tuple[int, int]]:
    from Skyjo.src.players.rl_player import RLPlayer

    game = SkyjoGame()
    rl_player = RLPlayer(0, "RL", model=model)
    opponent = PassivePlayer(1, "Opponent")
    game.add_player(rl_player)
    game.add_player(opponent)

    target_col = run_index % 4
    missing_row = run_index % 3
    missing_pos = (missing_row, target_col)

    values = [
        [3, 4, 5, 6],
        [7, 8, 9, 10],
        [-1, 0, 1, 2],
    ]
    for row in range(3):
        values[row][target_col] = 11 if row == missing_row else 12

    rl_player.player_state.grid = _revealed_grid(values)
    opponent.player_state.grid = _revealed_grid(
        [
            [-2, -1, 0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ]
    )

    game.game_state.discard_pile = [_revealed_card(12)]
    game.game_state.draw_pile = [Card(-2)]
    game.game_state.hand_card = None
    game.game_state.phase = TurnPhase.CHOOSE_DRAW
    return game, rl_player, missing_pos


def _assert_legal(action: Action, legal_actions: list[Action]) -> None:
    assert action in legal_actions, f"{action} not in legal actions {legal_actions}"


def test_rl_model_reports_whether_it_clears_available_column():
    """Diagnostic: print whether the checkpoint takes a discard that clears a column."""
    sb3_contrib = pytest.importorskip("sb3_contrib")

    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"RL checkpoint not found: {CHECKPOINT_PATH}")

    model = sb3_contrib.MaskablePPO.load(str(CHECKPOINT_PATH), device="cpu")

    attempts = []
    for run_index in range(10):
        game, rl_player, missing_pos = _make_column_clear_game(model, run_index)

        draw_legal = game.get_legal_actions(rl_player)
        assert Action(ActionType.DRAW_OPEN_CARD) in draw_legal

        draw_action = rl_player.select_action(
            game.get_observation(rl_player), draw_legal
        )
        _assert_legal(draw_action, draw_legal)

        swap_action = None
        column_cleared = False
        if draw_action.type == ActionType.DRAW_OPEN_CARD:
            game.execute_action(rl_player, draw_action)
            game.game_state.remove_unfiorm_columns_to_discard_pile(
                rl_player.player_state
            )

            swap_legal = game.get_legal_actions(rl_player)
            target_swap = Action(ActionType.SWAP_CARD, pos=missing_pos)
            assert target_swap in swap_legal

            swap_action = rl_player.select_action(
                game.get_observation(rl_player), swap_legal
            )
            _assert_legal(swap_action, swap_legal)

            game.execute_action(rl_player, swap_action)
            game.game_state.remove_unfiorm_columns_to_discard_pile(
                rl_player.player_state
            )
            column_cleared = len(rl_player.player_state.grid[0]) == 3

        attempts.append(
            {
                "run": run_index + 1,
                "missing_pos": missing_pos,
                "draw_action": draw_action,
                "swap_action": swap_action,
                "column_cleared": column_cleared,
            }
        )

    draw_open_count = sum(
        attempt["draw_action"].type == ActionType.DRAW_OPEN_CARD for attempt in attempts
    )
    target_swap_count = sum(
        attempt["swap_action"]
        == Action(ActionType.SWAP_CARD, pos=attempt["missing_pos"])
        for attempt in attempts
    )
    cleared_count = sum(attempt["column_cleared"] for attempt in attempts)

    print("\nRL column-clear diagnostic")
    print(f"draw_open={draw_open_count}/10")
    print(f"target_swap={target_swap_count}/10")
    print(f"column_cleared={cleared_count}/10")
    for attempt in attempts:
        print(
            "run={run} missing_pos={missing_pos} draw={draw_action} "
            "swap={swap_action} cleared={column_cleared}".format(**attempt)
        )
