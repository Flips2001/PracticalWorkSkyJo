"""Does integrated gradients explain the column clear the way a human would?

Run this file from a real terminal (the curses view needs one). It plays the
randomized column clear scenario (see ``column_clear_rate``) and, whenever the
agent takes the open 9 and swaps it onto the hidden slot of its 9-9-? column,
takes the explanations the UI shows for both decisions. A human explanation of
the clear rests on three cards: the 9 being taken (discard top, then hand card)
and the two 9s already in the column. The experiment reports how much of each
decision IG puts on those three cards, then shows each decision in the game
UI's analysis view with every attribution averaged over all cleared runs.
"""

import curses
import logging
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Skyjo.experiment.column_clear_rate import build_randomized_game  # noqa: E402
from Skyjo.scenarios.scenario import (  # noqa: E402
    AGENT_ID,
    AGENT_NAME,
    DEFAULT_MODEL,
    DEVICE,
    HUMAN_ID,
    _resolve_model_path,
)
from Skyjo.src.action import Action  # noqa: E402
from Skyjo.src.action_type import ActionType  # noqa: E402
from Skyjo.src.observation import ObservedCard  # noqa: E402
from Skyjo.src.players.rl_player import RLPlayer  # noqa: E402
from Skyjo.src.rl.integrated_gradients import ActionExplanation  # noqa: E402
from Skyjo.src.ui.terminal_ui import _ANALYSIS_COL, TerminalRenderer  # noqa: E402

RUNS = 1000
MODEL = DEFAULT_MODEL
VIEWER_NAME = "Phillips"
TOP_UNITS_SHOWN = 6
# The agent's 9-9-? column in the column clear scenario.
COLUMN_NINES = ((0, 1), (1, 1))
CLEAR_PATH = (
    Action(ActionType.DRAW_OPEN_CARD),
    Action(ActionType.SWAP_CARD, pos=(2, 1)),
)
# The 9 being taken is the discard top before the draw and the hand card after.
TAKEN_NINE = {ActionType.DRAW_OPEN_CARD: "discard", ActionType.SWAP_CARD: "hand"}
HIDDEN = ObservedCard(value=None, face_up=False)
# Screen size of one TerminalRenderer._render_analysis block.
ANALYSIS_WIDTH = 78
ANALYSIS_HEIGHT = 23


class _ExplanationRecorder:
    """Action hook that keeps (snapshot, explanation) for every agent action.

    The snapshot is the opponent's view, as TerminalGameUI captures it, so the
    renderer maps the explanation's "own" units onto the agent's grid.
    """

    def __init__(self, agent: RLPlayer):
        self.agent = agent
        self.decisions = []

    def before_action(self, game, player, action):
        # RLPlayer explains in select_action, which runs right before this hook.
        snapshot = game.get_observation(game.players[HUMAN_ID])
        self.decisions.append((snapshot, self.agent.last_explanation))

    def after_action(self, game, player, action):
        pass


def _unit_key(unit) -> str:
    """A name for the unit that is the same in every run, unlike its label."""
    if unit.group == "cell":
        return f"{unit.owner} R{unit.pos[0]}C{unit.pos[1]}"
    if unit.group == "discard_counts":
        return f"{unit.card_value}s in discard pile"
    return unit.group


def _evidence_keys(action_type: ActionType) -> set:
    own = {f"own R{r}C{c}" for r, c in COLUMN_NINES}
    return own | {TAKEN_NINE[action_type]}


def run_once(model: MaskablePPO):
    """The turn's (snapshot, explanation) pairs if it cleared, else None."""
    agent = RLPlayer(
        AGENT_ID, "RL", model=model, deterministic=True, explain_moves=True
    )
    game = build_randomized_game(agent)
    recorder = _ExplanationRecorder(agent)
    game.action_hooks = recorder
    game.turn(agent)
    actions = tuple(e.action for _, e in recorder.decisions)
    cleared = game.total_columns_cleared.get(AGENT_ID, 0) > 0
    return recorder.decisions if cleared and actions == CLEAR_PATH else None


def mean_explanation(explanations) -> ActionExplanation:
    """One explanation whose every attribution is the mean over ``explanations``."""
    # Units are matched by key: a run can lack one (e.g. no discard unit when the
    # taken 9 emptied the pile), which counts as 0 so the means stay complete.
    templates, sums = {}, defaultdict(float)
    for explanation in explanations:
        for unit in explanation.units:
            templates.setdefault(_unit_key(unit), unit)
            sums[_unit_key(unit)] += unit.attribution
    units = [
        replace(unit, attribution=sums[key] / len(explanations))
        for key, unit in templates.items()
    ]
    return replace(
        explanations[0],
        units=units,
        target_score=float(np.mean([e.target_score for e in explanations])),
        baseline_score=float(np.mean([e.baseline_score for e in explanations])),
    )


def mean_snapshot(snapshots):
    """The decision board as far as it is the same in every run.

    The opponent's grid and the discard top below the taken 9 are random per
    run, so they show face down with a score of 0; the agent's grid and score
    are fixed by the scenario; discard counts and draw pile are rounded means.
    """
    first = snapshots[0]
    rows, cols = len(first.card_grid), len(first.card_grid[0])

    def mean_ints(get):
        return tuple(
            int(round(v)) for v in np.mean([get(s) for s in snapshots], axis=0)
        )

    return replace(
        first,
        card_grid=tuple(tuple(HIDDEN for _ in range(cols)) for _ in range(rows)),
        discard_top=first.discard_top if first.hand_card is None else HIDDEN,
        discard_pile_value_counts=mean_ints(lambda s: s.discard_pile_value_counts),
        scores=tuple(0 if i == HUMAN_ID else v for i, v in enumerate(first.scores)),
        draw_pile_size=int(round(np.mean([s.draw_pile_size for s in snapshots]))),
    )


def report(action_type: ActionType, explanations) -> None:
    evidence = _evidence_keys(action_type)
    attributions = defaultdict(list)
    shares, top3_hits = [], 0
    for explanation in explanations:
        by_key = {_unit_key(u): u.attribution for u in explanation.units}
        for key, value in by_key.items():
            attributions[key].append(value)
        top3 = sorted(by_key, key=by_key.get, reverse=True)[:3]
        top3_hits += set(top3) == evidence
        # Completeness: attributions sum to the total influence, so this is the
        # fraction of the log-prob change IG credits to the three 9s.
        shares.append(sum(by_key[k] for k in evidence) / explanation.total_influence)

    totals = np.array([e.total_influence for e in explanations])
    print(f"\n{action_type}, {len(explanations)} decisions:")
    print(f"  total influence (log p - log p_baseline)  median {np.median(totals):.3f}")
    print(
        f"  the three 9s are the top 3 units          "
        f"{top3_hits / len(explanations):.1%}"
    )
    print(
        f"  share of total influence on the three 9s  median {np.median(shares):.1%}"
        f"  p10 {np.percentile(shares, 10):.1%}"
    )
    print("  mean attribution, strongest units and the three 9s:")
    means = {key: np.mean(values) for key, values in attributions.items()}
    ranked = sorted(means, key=lambda k: abs(means[k]), reverse=True)
    for i, key in enumerate(ranked):
        if i < TOP_UNITS_SHOWN or key in evidence:
            marker = "*" if key in evidence else " "
            print(f"   {marker} {key:<22}{means[key]:+.3f}")


def show_in_ui(stdscr, views) -> None:
    """Show every ``(action_text, explanation, snapshot)`` in the UI's analysis view.

    Side by side at the game's analysis column when the window is wide enough,
    stacked otherwise.
    """
    renderer = TerminalRenderer(stdscr)
    while True:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        side_by_side = cols >= _ANALYSIS_COL + ANALYSIS_WIDTH
        for i, (action_text, explanation, snapshot) in enumerate(views):
            top, left = (
                (1, 2 + i * (_ANALYSIS_COL - 2))
                if side_by_side
                else (1 + i * ANALYSIS_HEIGHT, 2)
            )
            renderer._render_analysis(
                top, left, action_text, explanation, snapshot, VIEWER_NAME, AGENT_NAME
            )
        renderer._safe_addstr(rows - 1, 2, "q quit  │  resize the window to relayout")
        stdscr.refresh()
        if stdscr.getch() in (ord("q"), ord("Q")):
            return


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    model = MaskablePPO.load(_resolve_model_path(MODEL), device=DEVICE)
    turns = [run_once(model) for _ in tqdm(range(RUNS))]
    cleared = [t for t in turns if t is not None]
    if not cleared:
        print(f"{MODEL}, {RUNS} runs, none cleared the column")
        return

    views = []
    for i, action in enumerate(CLEAR_PATH):
        snapshots, explanations = zip(*(turn[i] for turn in cleared))
        views.append(
            (
                f"{AGENT_NAME}: {action}  (mean over {len(cleared)} runs)",
                mean_explanation(explanations),
                mean_snapshot(snapshots),
            )
        )
    try:
        curses.wrapper(show_in_ui, views)
    except curses.error:
        print(
            "No terminal available for the curses UI view; run from a real "
            "terminal, or enable 'Emulate terminal in output console' in the "
            "PyCharm run configuration."
        )

    # After curses, so the summary stays on screen.
    print(f"{MODEL}, {RUNS} runs, {len(cleared)} cleared the column (* = a 9)")
    for i, action in enumerate(CLEAR_PATH):
        report(action.type, [turn[i][1] for turn in cleared])


if __name__ == "__main__":
    main()
