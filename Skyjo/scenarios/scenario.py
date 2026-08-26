"""Hand-built Skyjo positions for observing the trained agent's behavior.

A scenario is plain data: the human's grid, the agent's grid, the discard pile
and - optionally - the next cards off the draw pile. The shipped scenarios give
the first move to the agent and explain which policy behavior the position is
designed to expose. Running a scenario module deals exactly that position and
hands control to the terminal UI, so play continues from there instead of from a
fresh round.

Grid entries (three rows, up to four cards each):

    5       face-up 5
    "5?"    face-down card that will turn out to be a 5
    "?"     face-down card, value picked from whatever the deck has left

Rows shorter than four cards model a column that has already been cleared.
Every named card is taken out of one real 150 card Skyjo deck, so a scenario
that uses six -2s is rejected instead of quietly distorting the card counts the
agent reasons about.

Run a scenario with, for example::

    python -m Skyjo.scenarios.column_clear
"""

from __future__ import annotations

import argparse
import curses
import logging
import os
import random
import re
import textwrap
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

from sb3_contrib import MaskablePPO

from Skyjo.src.card import Card
from Skyjo.src.players.rl_player import RLPlayer
from Skyjo.src.players.terminal_player import TerminalPlayer
from Skyjo.src.rl.encoding import INITIAL_CARD_COUNTS
from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.turn_phase import TurnPhase
from Skyjo.src.ui.terminal_game_ui import TerminalGameUI

# SkyjoGame indexes seats by player id, so the agent is always seated first.
AGENT_ID = 0
HUMAN_ID = 1
AGENT_NAME = "RL Player"
HUMAN_NAME = "You"

GRID_ROWS = 3
GRID_COLS = 4

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "src", "rl", "checkpoints"
)
DEFAULT_MODEL = "skyjo_ppo_best"
DEVICE = "cpu"

CardSpec = Union[int, str]
GridSpec = Sequence[Sequence[CardSpec]]

# (value, face_up); value is None for a "?" card still to be picked.
_ParsedCard = Tuple[Optional[int], bool]


@dataclass(frozen=True)
class Scenario:
    """One hand-built board, played out as a single round from where it stands.

    Attributes:
        name: Short title shown on the briefing screen.
        description: Why the agent's behavior in this position is worth
            observing; blank lines separate paragraphs.
        your_grid: Your 3-row grid, see the module docstring for card entries.
        opponent_grid: The agent's 3-row grid, same notation.
        discard: Discard pile from bottom to top; the last value is the face-up
            top card, the only one either side may take.
        draw: The next cards off the draw pile, in the order they get drawn.
            Everything the scenario does not name is shuffled in behind them.
        first_player: ``"you"`` or ``"agent"``, whoever is to move. Agent-first
            is the default because scenarios are intended as policy probes.
        your_total_score: Points you carry in from earlier rounds.
        opponent_total_score: Points the agent carries in from earlier rounds.
        seed: Seeds the global RNG when the scenario is applied, which fixes the
            ``"?"`` values, the deck shuffle and everything the engine draws
            afterwards, so the same scenario plays out the same way twice.
    """

    name: str
    description: str
    your_grid: GridSpec
    opponent_grid: GridSpec
    discard: Sequence[int]
    draw: Sequence[int] = ()
    first_player: str = "agent"
    your_total_score: int = 0
    opponent_total_score: int = 0
    seed: Optional[int] = None


def apply_scenario(game: SkyjoGame, scenario: Scenario) -> None:
    """Replace the round ``game`` just dealt with the scenario's position.

    Both players must already be added, the agent first, and the game is left
    ready for ``play_round(setup_round=False)``.
    """
    if scenario.first_player not in ("you", "agent"):
        raise ValueError(
            f"first_player: expected 'you' or 'agent', got {scenario.first_player!r}"
        )
    if [player.player_id for player in game.players] != [AGENT_ID, HUMAN_ID]:
        raise ValueError(
            "a scenario needs exactly two players, the agent (id 0) added before "
            "you (id 1), because SkyjoGame indexes seats by player id"
        )

    if scenario.seed is not None:
        random.seed(scenario.seed)

    your_grid, opponent_grid, discard, draw_pile = _deal(scenario)

    game.player_states[HUMAN_ID].grid = your_grid
    game.player_states[AGENT_ID].grid = opponent_grid

    state = game.game_state
    state.discard_pile = discard
    state.draw_pile = draw_pile
    state.hand_card = None
    state.phase = TurnPhase.CHOOSE_DRAW
    state.current_player_id = HUMAN_ID if scenario.first_player == "you" else AGENT_ID
    # The starting flips already happened, in whatever way the position implies.
    state.round_start_flips = {player_id: 2 for player_id in game.player_states}
    state.final_turn_phase = False
    state.first_finisher_id = None
    state.players_to_finish = set()

    totals = [0, 0]
    totals[HUMAN_ID] = scenario.your_total_score
    totals[AGENT_ID] = scenario.opponent_total_score
    for player_id, total in enumerate(totals):
        game.player_states[player_id].set_final_game_score(total)
    state.all_player_final_scores = totals


def build_scenario_game(
    scenario: Scenario,
    agent,
    human,
    action_hooks=None,
) -> SkyjoGame:
    """Seat the two players agent-first and set the board to ``scenario``."""
    game = SkyjoGame(action_hooks=action_hooks)
    game.add_player(agent)
    game.add_player(human)
    apply_scenario(game, scenario)
    return game


def play_scenario(
    scenario: Scenario,
    model_path: Optional[str] = None,
    full_game: bool = False,
) -> None:
    """Play ``scenario`` in the terminal against the trained agent.

    Args:
        scenario: The position to play from.
        model_path: Checkpoint name or path for the agent.
        full_game: Keep playing normally dealt rounds after the scenario round,
            until one side reaches 100 points.
    """
    resolved = _resolve_model_path(model_path or DEFAULT_MODEL)
    print(f"Loading {os.path.basename(resolved)} ...")
    agent = RLPlayer(
        player_id=AGENT_ID,
        player_name=AGENT_NAME,
        model=MaskablePPO.load(resolved, device=DEVICE),
        # Always explain: analyze mode is toggled in-game ('a') and should show
        # the latest RL move the moment it is switched on.
        explain_moves=True,
    )

    try:
        curses.wrapper(
            lambda stdscr: play_in_terminal(stdscr, scenario, agent, full_game)
        )
    except curses.error:
        print("No terminal available for the curses UI; run from a real terminal.")


def run_scenario(scenario: Scenario, argv: Optional[Sequence[str]] = None) -> None:
    """Command line entry point for a scenario module."""
    parser = argparse.ArgumentParser(description=scenario.name)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Checkpoint name or path for the agent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the scenario seed for its '?' cards and deck shuffle.",
    )
    parser.add_argument(
        "--full-game",
        action="store_true",
        help="Keep playing after the scenario round, until someone reaches 100.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.CRITICAL)
    if args.seed is not None:
        scenario = replace(scenario, seed=args.seed)
    play_scenario(scenario, model_path=args.model, full_game=args.full_game)


def play_in_terminal(
    stdscr, scenario: Scenario, agent, full_game: bool = False
) -> None:
    """Show the briefing, then play ``scenario`` out in the curses UI.

    Args:
        stdscr: The curses screen to draw on.
        scenario: The position to play from.
        agent: The opponent, seated as player ``AGENT_ID``.
        full_game: Keep playing dealt rounds afterwards, until someone hits 100.
    """
    curses.curs_set(0)
    stdscr.keypad(True)

    ui = TerminalGameUI(
        stdscr=stdscr,
        player_id=HUMAN_ID,
        player_name=HUMAN_NAME,
        opponent_name=AGENT_NAME,
        analyze_mode=True,
    )
    human = TerminalPlayer(player_id=HUMAN_ID, player_name=HUMAN_NAME, ui=ui)
    game = build_scenario_game(scenario, agent, human, action_hooks=ui)

    def on_round_end(g: SkyjoGame) -> None:
        ui.show_round_summary(
            g.game_state.all_player_final_scores,
            [player.player_name for player in g.players],
            g.game_state.round_number - 1,
        )

    def on_game_over(g: SkyjoGame) -> None:
        ui.show_game_over(
            g.game_state.all_player_final_scores,
            [player.player_name for player in g.players],
        )

    try:
        _show_briefing(stdscr, scenario)
        _show_start_position(ui, game, human)

        if full_game:
            game.play_game(
                on_round_end=on_round_end,
                on_game_over=on_game_over,
                setup_first_round=False,
            )
            return

        game.play_round(setup_round=False)
        game.game_state.game_over()
        if game.game_state.is_game_over:
            on_game_over(game)
        else:
            on_round_end(game)
    except KeyboardInterrupt:
        pass


def _show_briefing(stdscr, scenario: Scenario) -> None:
    """Explain the position before the board takes over the screen."""
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()
    width = max(20, min(72, max_x - 4))

    lines = ["S C E N A R I O", scenario.name, ""]
    for paragraph in _paragraphs(scenario.description):
        lines.extend(textwrap.wrap(paragraph, width))
        lines.append("")
    lines += [
        f"You play '{HUMAN_NAME}' against '{AGENT_NAME}'"
        f"; {'you' if scenario.first_player == 'you' else 'the agent'} start.",
        "Analyze starts ON; Enter advances agent actions, a toggles it, q quits.",
        "",
        "Press any key to see the board ...",
    ]

    for row, line in enumerate(lines[: max_y - 2], start=1):
        attr = curses.A_BOLD if row <= 2 else curses.A_NORMAL
        try:
            stdscr.addstr(row, 2, line[: max_x - 3], attr)
        except curses.error:
            pass
    stdscr.refresh()
    stdscr.getch()


def _show_start_position(ui: TerminalGameUI, game: SkyjoGame, human) -> None:
    """Show the dealt position and hold there until the player is ready.

    Without this the board would first appear mid-turn whenever the scenario
    lets the agent move first.
    """
    ui.renderer.render_game(
        observation=game.get_observation(human),
        player_name=HUMAN_NAME,
        opponent_name=AGENT_NAME,
        legal_actions=[],
        selected_index=0,
        message="This is the scenario position. Press Enter to play from here.",
        show_actions=False,
        help_text=" Enter Start  │  q Quit ",
    )
    while True:
        key = ui.stdscr.getch()
        if key in (curses.KEY_ENTER, 10, 13):
            return
        if key in (ord("q"), ord("Q")):
            raise KeyboardInterrupt("Player quit the scenario")


def _paragraphs(description: str) -> List[str]:
    """Reflow a docstring-style description into single-line paragraphs."""
    blocks = re.split(r"\n\s*\n", textwrap.dedent(description).strip())
    return [" ".join(block.split()) for block in blocks if block.strip()]


def _resolve_model_path(name_or_path: str) -> str:
    """Resolve a checkpoint reference the same way the competitions do."""
    path = name_or_path
    if not (os.path.isabs(path) or os.path.exists(path)):
        path = os.path.join(CHECKPOINT_DIR, path)
    if not path.endswith(".zip"):
        path += ".zip"
    return path


def _deal(
    scenario: Scenario,
) -> Tuple[List[List[Card]], List[List[Card]], List[Card], List[Card]]:
    """Deal the scenario out of one full Skyjo deck.

    Returns:
        ``(your_grid, opponent_grid, discard_pile, draw_pile)``.
    """
    budget = dict(INITIAL_CARD_COUNTS)
    your_parsed = _parse_grid(scenario.your_grid, "your_grid")
    opponent_parsed = _parse_grid(scenario.opponent_grid, "opponent_grid")

    if not scenario.discard:
        raise ValueError("discard: a scenario needs at least one discarded card")
    discard: List[Card] = []
    for value in scenario.discard:
        _take(budget, value, "discard")
        discard.append(Card(value, face_up=True))

    top_of_deck: List[Card] = []
    for value in scenario.draw:
        _take(budget, value, "draw")
        top_of_deck.append(Card(value))

    # Named cards first, so the "?" wildcards can only use what is left over.
    for parsed, where in (
        (your_parsed, "your_grid"),
        (opponent_parsed, "opponent_grid"),
    ):
        for value, _ in (card for row in parsed for card in row):
            if value is not None:
                _take(budget, value, where)

    your_grid = _build_grid(your_parsed, budget, "your_grid")
    opponent_grid = _build_grid(opponent_parsed, budget, "opponent_grid")
    _reject_uniform_columns(your_grid, "your_grid")
    _reject_uniform_columns(opponent_grid, "opponent_grid")

    # draw_card() pops from the end, so the first named card goes on top.
    draw_pile = _remaining_cards(budget) + list(reversed(top_of_deck))
    return your_grid, opponent_grid, discard, draw_pile


def _parse_grid(grid: GridSpec, where: str) -> List[List[_ParsedCard]]:
    """Check the shape of a grid spec and parse every entry."""
    rows = [list(row) for row in grid]
    if len(rows) != GRID_ROWS:
        raise ValueError(f"{where}: needs {GRID_ROWS} rows, got {len(rows)}")

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(
            f"{where}: every row needs the same number of cards, got {sorted(widths)}"
        )
    width = widths.pop()
    if not 1 <= width <= GRID_COLS:
        raise ValueError(f"{where}: needs 1 to {GRID_COLS} cards per row, got {width}")

    return [[_parse_card(spec, where) for spec in row] for row in rows]


def _parse_card(spec: CardSpec, where: str) -> _ParsedCard:
    """Read one grid entry as ``(value, face_up)``; ``None`` value means "?"."""
    if isinstance(spec, int):
        return spec, True

    text = str(spec).strip()
    face_up = not text.endswith("?")
    if not face_up:
        text = text[:-1].strip()
    if not text:
        if face_up:
            raise ValueError(f"{where}: empty card entry {spec!r}")
        return None, False
    try:
        return int(text), face_up
    except ValueError:
        raise ValueError(
            f'{where}: cannot read card {spec!r}; write 5, "5", "5?" or "?"'
        ) from None


def _build_grid(
    parsed: List[List[_ParsedCard]], budget: Dict[int, int], where: str
) -> List[List[Card]]:
    """Turn parsed entries into cards, picking a value for every "?"."""
    return [
        [
            Card(
                value if value is not None else _pick_remaining_value(budget, where),
                face_up=face_up,
            )
            for value, face_up in row
        ]
        for row in parsed
    ]


def _take(budget: Dict[int, int], value: int, where: str) -> None:
    """Take one card of ``value`` out of the deck the scenario is dealt from."""
    if value not in budget:
        raise ValueError(f"{where}: {value} is not a Skyjo card value (-2 to 12)")
    if budget[value] < 1:
        raise ValueError(
            f"{where}: the deck holds {INITIAL_CARD_COUNTS[value]} cards of value "
            f"{value} and this scenario already used all of them"
        )
    budget[value] -= 1


def _pick_remaining_value(budget: Dict[int, int], where: str) -> int:
    """Pick a "?" value, weighted by what the deck still holds."""
    pool = [value for value, count in budget.items() for _ in range(count)]
    if not pool:
        raise ValueError(f"{where}: no cards left in the deck to fill a '?' with")
    value = random.choice(pool)
    budget[value] -= 1
    return value


def _remaining_cards(budget: Dict[int, int]) -> List[Card]:
    """Every card the scenario did not place, shuffled, face down."""
    cards = [Card(value) for value, count in budget.items() for _ in range(count)]
    random.shuffle(cards)
    return cards


def _reject_uniform_columns(grid: List[List[Card]], where: str) -> None:
    """A revealed uniform column would be swept away on the first action."""
    for col in range(len(grid[0])):
        column = [row[col] for row in grid]
        if not all(card.face_up for card in column):
            continue
        if len({card.get_value() for card in column}) == 1:
            raise ValueError(
                f"{where}: column {col} is already a complete set of "
                f"{column[0].get_value()}s, which the game clears on the first "
                "action; hide one of those cards or change a value"
            )
