"""ApiPlayer: delegates Skyjo decisions to https://skyjo.artzima.dev.

Uses the ``/api/session/infer-agent-step`` endpoint which operates at
*decision* granularity (CHOOSE_SOURCE → KEEP_OR_DISCARD → CHOOSE_POSITION)
via the server-side ``SkyjoDecisionEnv``.  This is essential because:

* The heuristic bots require ``decision_phase`` in the observation (only
  provided by ``SkyjoDecisionEnv.observe()``).
* MCTS models (baseline / belief) need the actual drawn-card value in the
  ``PendingTurnState`` for accurate forward simulations.

Each local sub-decision maps to one HTTP call.  The server returns an
updated ``decision_context`` that threads through the multi-step turn.
"""

from __future__ import annotations

import json
import logging
import random
import re
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

# Decision-phase IDs used by the remote API's SkyjoDecisionEnv.
_PHASE_SETUP_REVEAL = 0
_PHASE_CHOOSE_SOURCE = 1
_PHASE_KEEP_OR_DISCARD = 2
_PHASE_CHOOSE_POSITION = 3


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, timeout: int = _DEFAULT_TIMEOUT) -> dict:
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


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def _api_pos_to_local(api_pos: int) -> Tuple[int, int]:
    if not (0 <= api_pos < BOARD_SIZE):
        raise ValueError(f"API returned out-of-bounds position {api_pos}")
    return api_pos // COLS, api_pos % COLS


def _card_grid_to_api_board(
    card_grid: Optional[List[List[Optional[Card]]]],
) -> dict:
    """Convert a local 3xN grid to the API's 12-slot board dict."""
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


def _parse_pos(step_log: str) -> int:
    """Extract the trailing integer position from a step_log string."""
    match = re.search(r"(\d+)\s*$", step_log)
    if match:
        return int(match.group(1))
    raise ValueError(f"No position found in step_log: {step_log!r}")


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------


def _make_agent_config(
    agent_type: str,
    simulations: int,
    heuristic_bot_name: str,
    epsilon: float = 0.0,
) -> dict:
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"agent_type must be one of {AGENT_TYPES}, got {agent_type!r}")
    return {
        "type": agent_type,
        "checkpoint_path": None,
        "simulations": int(simulations),
        "device": "cpu",
        "ablate_belief_head": False,
        "heuristic_bot_name": heuristic_bot_name,
        "heuristic_bot_epsilon": float(epsilon),
    }


# ---------------------------------------------------------------------------
# ApiPlayer
# ---------------------------------------------------------------------------


class ApiPlayer(Player):
    """A Player that delegates decisions to the skyjo.artzima.dev backend.

    Uses ``/api/session/infer-agent-step`` for each sub-decision so the
    server-side ``SkyjoDecisionEnv`` provides proper ``decision_phase``
    observations and accounts for actual drawn-card values.

    Falls back to a random legal action on any API/mapping error.
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
        flip_heuristic_bot_name: str = "column_hunter",
        request_timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(player_id, player_name)
        self._base_url = base_url.rstrip("/")
        self._timeout = int(request_timeout)
        self._agent_main = _make_agent_config(
            agent_type, simulations, heuristic_bot_name, heuristic_epsilon
        )
        # Flip-only decisions use column_hunter for heuristic mode
        # (greedy_value_replacement 500s on the SETUP flip path).
        flip_bot = (
            flip_heuristic_bot_name if agent_type == "heuristic" else heuristic_bot_name
        )
        self._agent_flip = _make_agent_config(agent_type, simulations, flip_bot, 0.0)

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
        except Exception as exc:
            logger.warning("ApiPlayer falling back to random: %s", exc)
            return random.choice(legal_actions)

    # ------------------------------------------------------------------
    # Per-phase dispatching
    # ------------------------------------------------------------------

    def _select(self, observation: Observation, legal_actions: List[Action]) -> Action:
        phase = observation.turn_phase
        state = self._build_state(observation)

        if phase == TurnPhase.STARTING_FLIPS:
            return self._handle_setup_reveal(state, observation)

        if phase == TurnPhase.CHOOSE_DRAW:
            return self._handle_choose_draw(state, observation, legal_actions)

        if phase == TurnPhase.HAVE_DRAWN_OPEN:
            # Drew from discard – must swap into grid.
            drawn_value = observation.hand_card.value if observation.hand_card else 0
            return self._handle_choose_position(
                state,
                observation,
                source="DISCARD",
                drawn_value=drawn_value,
                keep=True,
            )

        if phase == TurnPhase.HAVE_DRAWN_HIDDEN:
            return self._handle_keep_or_discard(state, observation, legal_actions)

        if phase == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD:
            # The drawn card was discarded; flip a hidden card.
            drawn_value = (
                observation.discard_top.value if observation.discard_top else 0
            )
            return self._handle_choose_position(
                state,
                observation,
                source="DECK",
                drawn_value=drawn_value,
                keep=False,
            )

        return random.choice(legal_actions)

    # ------------------------------------------------------------------
    # Per-phase handlers
    # ------------------------------------------------------------------

    def _handle_setup_reveal(self, state: dict, observation: Observation) -> Action:
        ctx = self._make_context(observation.player_id, _PHASE_SETUP_REVEAL)
        resp = self._infer_step(state, ctx, agent=self._agent_flip)
        row, col = _api_pos_to_local(_parse_pos(resp["step_log"]))
        return Action(ActionType.FLIP_CARD, pos=(row, col))

    def _handle_choose_draw(
        self,
        state: dict,
        observation: Observation,
        legal_actions: List[Action],
    ) -> Action:
        ctx = self._make_context(observation.player_id, _PHASE_CHOOSE_SOURCE)
        resp = self._infer_step(state, ctx)
        if "discard" in resp["step_log"]:
            return Action(ActionType.DRAW_OPEN_CARD)
        return Action(ActionType.DRAW_HIDDEN_CARD)

    def _handle_keep_or_discard(
        self,
        state: dict,
        observation: Observation,
        legal_actions: List[Action],
    ) -> Action:
        drawn_value = observation.hand_card.value if observation.hand_card else None
        ctx = self._make_context(
            observation.player_id,
            _PHASE_KEEP_OR_DISCARD,
            source="DECK",
            drawn_value=drawn_value,
        )
        resp = self._infer_step(state, ctx)

        if "keep" in resp["step_log"]:
            # Agent wants to keep – need CHOOSE_POSITION step for swap target.
            # Use the UPDATED state from the response so the server sees
            # the drawn card in pending correctly.
            next_ctx = resp.get("decision_context")
            updated_state = resp.get("state", state)
            if next_ctx is not None:
                resp2 = self._infer_step(updated_state, next_ctx)
                row, col = _api_pos_to_local(_parse_pos(resp2["step_log"]))
                return Action(ActionType.SWAP_CARD, pos=(row, col))
            # Fallback: pick first legal SWAP_CARD.
            for a in legal_actions:
                if a.type == ActionType.SWAP_CARD:
                    return a
            return legal_actions[0]

        # Agent wants to discard.
        for a in legal_actions:
            if a.type == ActionType.DISCARD_CARD:
                return a
        return legal_actions[0]

    def _handle_choose_position(
        self,
        state: dict,
        observation: Observation,
        source: str,
        drawn_value: Optional[int],
        keep: bool,
    ) -> Action:
        ctx = self._make_context(
            observation.player_id,
            _PHASE_CHOOSE_POSITION,
            source=source,
            drawn_value=drawn_value,
            keep=keep,
        )
        agent = self._agent_main if keep else self._agent_flip
        resp = self._infer_step(state, ctx, agent=agent)
        pos = _parse_pos(resp["step_log"])
        row, col = _api_pos_to_local(pos)
        action_type = ActionType.SWAP_CARD if keep else ActionType.FLIP_CARD
        return Action(action_type, pos=(row, col))

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _infer_step(
        self,
        state: dict,
        decision_context: dict,
        agent: Optional[dict] = None,
    ) -> dict:
        payload = {
            "state": state,
            "agent": agent if agent is not None else self._agent_main,
            "decision_context": decision_context,
        }
        return _post_json(
            f"{self._base_url}/api/session/infer-agent-step",
            payload,
            timeout=self._timeout,
        )

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_state(self, observation: Observation) -> dict:
        num_players = max(1, len(observation.opponent_cards))

        boards = []
        for pid in range(num_players):
            if pid == observation.player_id:
                boards.append(_card_grid_to_api_board(observation.card_grid))
            else:
                boards.append(_card_grid_to_api_board(observation.opponent_cards[pid]))

        phase_str = (
            "SETUP" if observation.turn_phase == TurnPhase.STARTING_FLIPS else "MAIN"
        )

        setup_reveals = [0] * num_players
        if observation.turn_phase == TurnPhase.STARTING_FLIPS:
            setup_reveals[observation.player_id] = 1

        discard_pile: List[int] = []
        if observation.discard_top is not None:
            discard_pile = [int(observation.discard_top.value)]

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

        deck = [0] * max(0, int(observation.draw_pile_size))

        return {
            "initial_seed": 0,
            "rng_state_b64": "",
            "num_players": num_players,
            "history_window_k": 16,
            "score_limit": 100,
            "setup_mode": "auto",
            "manual_initial_reveals": True,
            "round_index": 1,
            "global_step": 0,
            "turns_in_round": 0,
            "phase": phase_str,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_context(
        actor: int,
        phase_id: int,
        source: Optional[str] = None,
        drawn_value: Optional[int] = None,
        keep: Optional[bool] = None,
    ) -> dict:
        return {
            "actor_player": actor,
            "decision_phase_id": phase_id,
            "pending_source": source,
            "pending_drawn_value": drawn_value,
            "pending_keep_drawn": keep,
        }
