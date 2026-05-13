"""ApiPlayer: delegates Skyjo decisions to https://skyjo.artzima.dev."""

from __future__ import annotations

import json
import logging
import random
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.observation import Observation
from Skyjo.src.players.player import Player
from Skyjo.src.turn_phase import TurnPhase

logger = logging.getLogger(__name__)

_API_BASE_URL = "https://skyjo.artzima.dev"
_DEFAULT_TIMEOUT = 15

AGENT_TYPES = ("heuristic", "baseline", "belief")
HEURISTIC_BOTS = (
    "greedy_value_replacement",
    "column_hunter",
    "information_first_flip",
)

ROWS = 3
COLS = 4
BOARD_SIZE = ROWS * COLS  # 12

_TAKE_DISCARD_BASE = 0
_DRAW_DECK_KEEP_BASE = 12
_DRAW_DECK_DISCARD_FLIP_BASE = 24
_SETUP_FLIP_BASE = 36


def _post_json(url: str, payload: dict, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """POST a JSON payload and return the decoded JSON response.

    The remote service is fronted by a CDN that 403s requests without a
    browser-like User-Agent and Origin/Referer header pair.
    """
    data = json.dumps(payload).encode("utf-8")
    origin = url.split("/api/")[0]
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Origin": origin,
            "Referer": f"{origin}/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/142.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:300]}") from exc


def _api_pos_to_local(api_pos: int) -> Tuple[int, int]:
    if not (0 <= api_pos < BOARD_SIZE):
        raise ValueError(f"API returned out-of-bounds position {api_pos}")
    return api_pos // COLS, api_pos % COLS


def _card_grid_to_api_board(
    card_grid: Optional[List[List[Optional[Card]]]],
) -> dict:
    """Convert a (possibly shrunken) local 3xN grid to the API's 12-slot board.

    Cleared columns are presented as trailing ``removed=True`` slots so the
    (row, col) -> api_pos mapping stays consistent. Hidden cards still get
    their underlying values sent (the server masks them via ``visible``).
    """
    cards = [0] * BOARD_SIZE
    visible = [False] * BOARD_SIZE
    removed = [False] * BOARD_SIZE
    if card_grid is None:
        return {"cards": cards, "visible": visible, "removed": removed}

    for row_idx in range(ROWS):
        row_cards = card_grid[row_idx] if row_idx < len(card_grid) else []
        for col_idx in range(COLS):
            api_pos = row_idx * COLS + col_idx
            if col_idx >= len(row_cards) or row_cards[col_idx] is None:
                removed[api_pos] = True
                continue
            card = row_cards[col_idx]
            cards[api_pos] = int(card.value)
            visible[api_pos] = bool(card.face_up)
    return {"cards": cards, "visible": visible, "removed": removed}


def _decode_action(action_id: int) -> Tuple[str, int]:
    if _TAKE_DISCARD_BASE <= action_id < _TAKE_DISCARD_BASE + BOARD_SIZE:
        return "TAKE_DISCARD_AND_REPLACE", action_id - _TAKE_DISCARD_BASE
    if _DRAW_DECK_KEEP_BASE <= action_id < _DRAW_DECK_KEEP_BASE + BOARD_SIZE:
        return "DRAW_DECK_KEEP_AND_REPLACE", action_id - _DRAW_DECK_KEEP_BASE
    if (
        _DRAW_DECK_DISCARD_FLIP_BASE
        <= action_id
        < _DRAW_DECK_DISCARD_FLIP_BASE + BOARD_SIZE
    ):
        return "DRAW_DECK_DISCARD_AND_FLIP", action_id - _DRAW_DECK_DISCARD_FLIP_BASE
    if _SETUP_FLIP_BASE <= action_id < _SETUP_FLIP_BASE + BOARD_SIZE:
        return "SETUP_FLIP", action_id - _SETUP_FLIP_BASE
    raise ValueError(f"Invalid action id from API: {action_id}")


def _make_agent_config(
    agent_type: str,
    simulations: int,
    heuristic_bot_name: str,
    epsilon: float = 0.0,
) -> dict:
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"agent_type must be one of {AGENT_TYPES}, got {agent_type!r}")
    if heuristic_bot_name not in HEURISTIC_BOTS:
        raise ValueError(
            f"heuristic_bot_name must be one of {HEURISTIC_BOTS}, "
            f"got {heuristic_bot_name!r}"
        )
    return {
        "type": agent_type,
        "checkpoint_path": None,
        "simulations": int(simulations),
        "device": "cpu",
        "ablate_belief_head": False,
        "heuristic_bot_name": heuristic_bot_name,
        "heuristic_bot_epsilon": float(epsilon),
    }


class _TurnPlan:
    """A planned full turn returned by the remote API."""

    __slots__ = ("macro", "position")

    def __init__(self, macro: str, position: int) -> None:
        self.macro = macro
        self.position = position

    def __repr__(self) -> str:  # pragma: no cover
        return f"_TurnPlan({self.macro}, pos={self.position})"


class ApiPlayer(Player):
    """A Player that delegates decisions to the skyjo.artzima.dev backend.

    Each *full local turn* triggers exactly one HTTP call (one extra call
    per STARTING_FLIPS reveal). The remote ``/api/session/infer-action``
    endpoint returns a single full-turn macro action; we cache the result
    and replay it across the local sub-decision callbacks.

    Any error (HTTP failure, malformed response, illegal mapped action)
    transparently degrades to a random legal action so the calling game
    loop never crashes.
    """

    def __init__(
        self,
        player_id: int,
        player_name: str,
        agent_type: str = "heuristic",
        simulations: int = 32,
        base_url: str = _API_BASE_URL,
        heuristic_bot_name: str = "greedy_value_replacement",
        heuristic_epsilon: float = 0.0,
        # column_hunter is used for flip-only decisions because the
        # remote greedy_value_replacement returns 500s on the SETUP /
        # flip-after-discard paths.
        flip_heuristic_bot_name: str = "column_hunter",
        request_timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(player_id, player_name)
        self._base_url = base_url.rstrip("/")
        self._timeout = int(request_timeout)
        self._agent_main = _make_agent_config(
            agent_type, simulations, heuristic_bot_name, heuristic_epsilon
        )
        flip_bot = (
            flip_heuristic_bot_name if agent_type == "heuristic" else heuristic_bot_name
        )
        self._agent_flip = _make_agent_config(agent_type, simulations, flip_bot, 0.0)
        self._plan: Optional[_TurnPlan] = None

    # ------------------------------------------------------------------
    # Public Player interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        if not legal_actions:
            raise ValueError("ApiPlayer.select_action called with no legal actions")
        try:
            return self._select(observation, legal_actions)
        except Exception as exc:  # pragma: no cover - network resilience
            logger.warning("ApiPlayer falling back to random legal action: %s", exc)
            self._plan = None
            return random.choice(legal_actions)

    # ------------------------------------------------------------------
    # Per-phase dispatching
    # ------------------------------------------------------------------

    def _select(self, observation: Observation, legal_actions: List[Action]) -> Action:
        phase = observation.turn_phase

        if phase == TurnPhase.STARTING_FLIPS:
            self._plan = None
            return self._setup_flip_action(observation, legal_actions)

        if phase == TurnPhase.CHOOSE_DRAW:
            self._plan = self._fetch_turn_plan(observation, legal_actions)
            return self._action_for_choose_draw(self._plan, legal_actions)

        if phase == TurnPhase.HAVE_DRAWN_HIDDEN:
            return self._action_for_have_drawn_hidden(legal_actions)

        if phase == TurnPhase.HAVE_DRAWN_OPEN:
            return self._action_for_have_drawn_open(legal_actions)

        if phase == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD:
            return self._action_for_flip_after_discard(legal_actions)

        return legal_actions[0]

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _fetch_turn_plan(
        self, observation: Observation, legal_actions: List[Action]
    ) -> _TurnPlan:
        state = self._build_state(observation, phase="MAIN")
        resp = self._infer_action(state, self._agent_main)
        ai = resp.get("ai_action") or {}
        action_id = int(ai["action_id"])
        macro, pos = _decode_action(action_id)
        plan = _TurnPlan(macro=macro, position=pos)
        if not self._plan_is_locally_feasible(plan, legal_actions):
            raise RuntimeError(
                f"API plan {plan} not compatible with legal actions {legal_actions}"
            )
        return plan

    def _setup_flip_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        state = self._build_state(observation, phase="SETUP")
        resp = self._infer_action(state, self._agent_flip)
        ai = resp.get("ai_action") or {}
        action_id = int(ai["action_id"])
        macro, api_pos = _decode_action(action_id)
        if macro != "SETUP_FLIP":
            raise RuntimeError(
                f"Expected SETUP_FLIP from API in setup phase, got {macro}"
            )
        return self._action_with_pos(
            ActionType.FLIP_CARD, api_pos, legal_actions, ActionType.FLIP_CARD
        )

    def _infer_action(self, state: dict, agent: dict) -> dict:
        payload = {"state": state, "agent": agent}
        return _post_json(
            f"{self._base_url}/api/session/infer-action",
            payload,
            timeout=self._timeout,
        )

    # ------------------------------------------------------------------
    # Plan -> local Action translation
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_is_locally_feasible(plan: _TurnPlan, legal_actions: List[Action]) -> bool:
        macro = plan.macro
        if macro == "TAKE_DISCARD_AND_REPLACE":
            return any(a.type == ActionType.DRAW_OPEN_CARD for a in legal_actions)
        if macro in ("DRAW_DECK_KEEP_AND_REPLACE", "DRAW_DECK_DISCARD_AND_FLIP"):
            return any(a.type == ActionType.DRAW_HIDDEN_CARD for a in legal_actions)
        return False

    @staticmethod
    def _action_for_choose_draw(plan: _TurnPlan, legal_actions: List[Action]) -> Action:
        if plan.macro == "TAKE_DISCARD_AND_REPLACE":
            target_type = ActionType.DRAW_OPEN_CARD
        else:
            target_type = ActionType.DRAW_HIDDEN_CARD
        for a in legal_actions:
            if a.type == target_type:
                return a
        return legal_actions[0]

    def _action_for_have_drawn_hidden(self, legal_actions: List[Action]) -> Action:
        plan = self._plan
        if plan is None:
            return random.choice(legal_actions)
        if plan.macro == "DRAW_DECK_KEEP_AND_REPLACE":
            return self._action_with_pos(
                ActionType.SWAP_CARD,
                plan.position,
                legal_actions,
                ActionType.SWAP_CARD,
            )
        if plan.macro == "DRAW_DECK_DISCARD_AND_FLIP":
            for a in legal_actions:
                if a.type == ActionType.DISCARD_CARD:
                    return a
        for a in legal_actions:
            if a.type == ActionType.SWAP_CARD:
                return a
        return legal_actions[0]

    def _action_for_have_drawn_open(self, legal_actions: List[Action]) -> Action:
        plan = self._plan
        swap_actions = [a for a in legal_actions if a.type == ActionType.SWAP_CARD]
        if plan is None or plan.macro != "TAKE_DISCARD_AND_REPLACE":
            return random.choice(swap_actions or legal_actions)
        return self._action_with_pos(
            ActionType.SWAP_CARD,
            plan.position,
            legal_actions,
            ActionType.SWAP_CARD,
        )

    def _action_for_flip_after_discard(self, legal_actions: List[Action]) -> Action:
        plan = self._plan
        if plan is None or plan.macro != "DRAW_DECK_DISCARD_AND_FLIP":
            return random.choice(legal_actions)
        return self._action_with_pos(
            ActionType.FLIP_CARD,
            plan.position,
            legal_actions,
            ActionType.FLIP_CARD,
        )

    @staticmethod
    def _action_with_pos(
        action_type: ActionType,
        api_pos: int,
        legal_actions: List[Action],
        fallback_type: ActionType,
    ) -> Action:
        target = _api_pos_to_local(api_pos)
        for a in legal_actions:
            if a.type == action_type and a.pos == target:
                return a
        candidates = [a for a in legal_actions if a.type == fallback_type]
        if candidates:
            return random.choice(candidates)
        return random.choice(legal_actions)

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_state(self, observation: Observation, phase: str) -> dict:
        num_players = max(1, len(observation.opponent_cards))

        boards = []
        for pid in range(num_players):
            if pid == observation.player_id:
                boards.append(_card_grid_to_api_board(observation.card_grid))
            else:
                boards.append(_card_grid_to_api_board(observation.opponent_cards[pid]))

        # The server only consults setup_reveals_remaining when phase is
        # SETUP. Mark only the active player as needing one reveal so its
        # legal-action set is the unrevealed positions on its own board.
        setup_reveals = [0] * num_players
        if phase == "SETUP":
            setup_reveals[observation.player_id] = 1

        # Server agents only ever peek at discard_pile[-1].
        discard_pile: List[int] = []
        if observation.discard_top is not None:
            discard_pile = [int(observation.discard_top.value)]

        # Final-turn / round_ender bookkeeping.
        pending_final: List[int] = []
        round_ender: Optional[int] = None
        if observation.final_turn_phase and observation.first_finisher_id is not None:
            round_ender = int(observation.first_finisher_id)
            pending_final = [
                pid
                for pid in range(num_players)
                if pid != observation.first_finisher_id
            ]

        total_scores = list(observation.total_scores or [0] * num_players)
        round_scores = list(observation.scores or [0] * num_players)
        total_scores = (total_scores + [0] * num_players)[:num_players]
        round_scores = (round_scores + [0] * num_players)[:num_players]

        # Deck card values are not used by the server agents; only the
        # length acts as a coarse round-progression signal.
        deck = [0] * max(0, int(observation.draw_pile_size))

        return {
            "initial_seed": 0,
            # Empty -> server skips rng.setstate(), avoiding pickle
            # / numpy version mismatches.
            "rng_state_b64": "",
            "num_players": num_players,
            "history_window_k": 16,
            "score_limit": 100,
            "setup_mode": "auto",
            # We always supply boards / setup_reveals explicitly.
            "manual_initial_reveals": True,
            "round_index": 1,
            "global_step": 0,
            "turns_in_round": 0,
            "phase": phase,  # "SETUP" or "MAIN"
            "current_player": int(observation.player_id),
            "scores": total_scores,
            "round_scores": round_scores,
            "deck": deck,
            "discard_pile": discard_pile,
            "boards": boards,
            "setup_reveals_remaining": setup_reveals,
            "pending_final_turn_players": pending_final,
            "round_ender": round_ender,
            "column_clear_used_this_round": [False] * num_players,
            "game_over": False,
            "round_history_start_index": 0,
            "public_history": [],
        }
