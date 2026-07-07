import curses
from unittest.mock import MagicMock, patch

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.observation import Observation
from Skyjo.src.rl.integrated_gradients import ActionExplanation, UnitAttribution
from Skyjo.src.turn_phase import TurnPhase
from Skyjo.src.ui.terminal_ui import HEAT_PAIR_BASE, TerminalRenderer, heat_attr


def _pair(n: int) -> int:
    return n << 8


def _curses_patches():
    return (
        patch("curses.start_color"),
        patch("curses.use_default_colors"),
        patch("curses.init_pair"),
        patch("curses.curs_set"),
        patch("curses.color_pair", side_effect=_pair),
        patch.object(curses, "COLORS", 256, create=True),
    )


def _hidden_grid():
    return [[Card(0, face_up=False) for _ in range(4)] for _ in range(3)]


def _observation(top_left_value=None):
    grid = _hidden_grid()
    if top_left_value is not None:
        grid[0][0] = Card(top_left_value, face_up=True)
    return Observation(
        player_id=0,
        card_grid=grid,
        hand_card=None,
        opponent_cards=[None, _hidden_grid()],
        scores=[12, 0],
        discard_top=Card(5, face_up=True),
        draw_pile_size=100,
        turn_phase=TurnPhase.CHOOSE_DRAW,
        draw_pile_value_counts=[5] + [10] * 14,
        total_scores=[3, 7],
    )


def _explanation(scale=1.0):
    # The explanation belongs to the RL opponent: "opponent" units describe
    # the viewer's grid, "own" units the opponent's grid.
    units = [
        UnitAttribution(
            "your 12 at R0C0", 0.9 * scale, group="cell", owner="opponent", pos=(0, 0)
        ),
        UnitAttribution("discard 5", 0.6 * scale, group="discard"),
        UnitAttribution(
            "remaining -2s in deck", -0.3 * scale, group="deck", card_value=-2
        ),
        UnitAttribution(
            "your hidden card at R1C1",
            0.01 * scale,
            group="cell",
            owner="opponent",
            pos=(1, 1),
        ),
    ]
    return ActionExplanation(
        action=Action(ActionType.DRAW_OPEN_CARD),
        action_index=1,
        units=units,
        target_score=-0.2 * scale,
        baseline_score=-1.4 * scale,
    )


def _render(explanation, snapshot=None):
    stdscr = MagicMock()
    stdscr.getmaxyx.return_value = (60, 200)
    patches = _curses_patches()
    for p in patches:
        p.start()
    try:
        renderer = TerminalRenderer(stdscr)
        renderer.render_game(
            # The live board moved on: the 12 the explanation refers to only
            # exists in the snapshot.
            observation=_observation(top_left_value=3),
            player_name="You",
            opponent_name="RL",
            legal_actions=[],
            selected_index=0,
            opponent_last_action="RL: draw open card" if explanation else "",
            opponent_explanation=explanation,
            opponent_snapshot=snapshot,
            show_actions=False,
        )
    finally:
        for p in patches:
            p.stop()
    return [call.args for call in stdscr.addstr.call_args_list]


def _attrs_of(calls, text):
    return [attr for _, _, rendered, attr in calls if rendered == text]


def test_heat_attr_buckets():
    patches = _curses_patches()
    for p in patches:
        p.start()
    try:
        assert heat_attr(0.0) == 0
        assert heat_attr(0.01) == _pair(HEAT_PAIR_BASE)
        assert heat_attr(1.0) == _pair(HEAT_PAIR_BASE + 5)
    finally:
        for p in patches:
            p.stop()


def test_heatmap_is_painted_on_snapshot_not_live_board():
    calls = _render(_explanation(), snapshot=_observation(top_left_value=12))

    # The snapshot's 12 (strongest unit) gets the hottest tint; the live
    # board's 3 at the same position keeps its plain value color.
    assert _attrs_of(calls, " 12 ") == [_pair(HEAT_PAIR_BASE + 5) | curses.A_BOLD]
    assert _attrs_of(calls, "  3 ") == [_pair(2) | curses.A_BOLD]
    # Discard is rendered in both blocks: live plain, snapshot tinted.
    assert _attrs_of(calls, "[5]") == [
        _pair(3) | curses.A_BOLD,
        _pair(HEAT_PAIR_BASE + 4) | curses.A_BOLD,
    ]
    # Deck -2 count: live plain, snapshot tinted (0.3/0.9 → level 2). The
    # dim entries are the value-header "5" of each block's deck panel.
    assert _attrs_of(calls, "  5") == [
        _pair(0),
        curses.A_DIM,
        _pair(HEAT_PAIR_BASE + 2),
        curses.A_DIM,
    ]

    texts = [rendered for _, _, rendered, _ in calls]
    assert texts.count("Deck:") == 2
    assert any(text.startswith("influence:") for text in texts)
    assert not any(text.startswith(("toward ", "against ")) for text in texts)


def test_render_game_without_explanation_has_no_analysis_block():
    calls = _render(None)

    texts = [rendered for _, _, rendered, _ in calls]
    assert "Integrated Gradients" not in texts
    assert texts.count("Deck:") == 1
    assert _attrs_of(calls, "  3 ") == [_pair(2) | curses.A_BOLD]


def test_faint_units_get_the_lowest_tint():
    calls = _render(_explanation(), snapshot=_observation(top_left_value=12))

    attrs = _attrs_of(calls, " ?? ")
    # 0.01/0.9 lands in the lowest bucket: the faint unit is tinted green.
    assert _pair(HEAT_PAIR_BASE) | curses.A_BOLD in attrs
    # Cells without any unit keep their base color.
    assert _pair(5) | curses.A_BOLD in attrs


def test_tints_are_relative_to_each_move():
    # A move with uniformly tiny attributions shows the same relative tints
    # as a strong one: the heatmap always ranks within the move.
    calls = _render(_explanation(scale=0.01), snapshot=_observation(top_left_value=12))

    assert _attrs_of(calls, " 12 ") == [_pair(HEAT_PAIR_BASE + 5) | curses.A_BOLD]
