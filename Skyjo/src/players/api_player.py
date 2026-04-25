import json
import random
import re
import urllib.request
from typing import List, Optional

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.observation import Observation
from Skyjo.src.players.player import Player
from Skyjo.src.turn_phase import TurnPhase

_API_BASE_URL = "https://skyjo.artzima.dev"

# Valid agent type strings accepted by the remote API.
AGENT_TYPES = ("heuristic", "baseline", "belief")


def _make_agent_config(agent_type: str, simulations: int = 32) -> dict:
    """
    Build the agent config dict for the given type.
    agent_type: "heuristic" | "baseline" | "belief"
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"agent_type must be one of {AGENT_TYPES}, got {agent_type!r}")
    base = {
        "type": agent_type,
        "checkpoint_path": None,
        "simulations": simulations,
        "device": "cpu",
        "ablate_belief_head": False,
        "heuristic_bot_name": "greedy_value_replacement",
        "heuristic_bot_epsilon": 0.0,
    }
    return base


def _make_flip_agent_config(agent_type: str, simulations: int = 32) -> dict:
    """
    Agent config to use for flip decisions (setup reveal / flip-after-discard).
    For heuristic agents, column_hunter is used because greedy_value_replacement
    crashes with a 500 on that decision path. For model-based agents the same
    config is used for all decisions.
    """
    cfg = _make_agent_config(agent_type, simulations)
    if agent_type == "heuristic":
        cfg["heuristic_bot_name"] = "column_hunter"
    return cfg


# Decision phase IDs as used by the remote API's SkyjoDecisionEnv.
_PHASE_SETUP_REVEAL = 0
_PHASE_CHOOSE_SOURCE = 1
_PHASE_KEEP_OR_DISCARD = 2
_PHASE_CHOOSE_POSITION = 3


def _make_rng_state_b64() -> str:
    # The server skips setstate when this string is empty (see restore_env_state).
    # Sending our local numpy state would cause a version mismatch error.
    return ""


def _api_pos_to_local(api_pos: int):
    """Convert row-major API position (0-11) to local (row, col). Layout: pos = row*4 + col."""
    if not (0 <= api_pos <= 11):
        raise ValueError(f"API returned out-of-bounds position {api_pos}")
    return api_pos // 4, api_pos % 4


def _card_grid_to_api_board(card_grid: List[List[Optional[Card]]]) -> dict:
    """
    Convert a 3×N local card grid to the API's board dict (row-major, 12 positions).
    Columns that have been cleared from the local grid are marked removed=True.
    Hidden card values use 0 as placeholder; the heuristic only observes visible values.
    """
    cards = [0] * 12
    visible = [False] * 12
    removed = [False] * 12
    for row in range(3):
        row_cards = card_grid[row] if card_grid is not None else []
        for col in range(4):
            api_pos = row * 4 + col
            if col >= len(row_cards):
                removed[api_pos] = True
            else:
                card = row_cards[col]
                if card is not None and card.face_up:
                    cards[api_pos] = card.value
                    visible[api_pos] = True
    return {"cards": cards, "visible": visible, "removed": removed}


def _post_json(url: str, payload: dict, timeout: int = 10) -> dict:
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class ApiPlayer(Player):
    """
    A Skyjo player that delegates every decision to the remote AI API at
    https://skyjo.artzima.dev using the greedy_value_replacement heuristic.

    Each call to select_action reconstructs a minimal game state from the
    current observation, maps the local turn phase to the API's decision
    phase, and issues one or two HTTP calls to /api/session/infer-agent-step.
    Falls back to a random legal action if the API is unreachable or returns
    an unexpected response.
    """

    def __init__(
        self,
        player_id: int,
        player_name: str,
        agent_type: str = "heuristic",
        simulations: int = 32,
        base_url: str = _API_BASE_URL,
    ):
        """
        agent_type: "heuristic" (greedy_value_replacement),
                    "baseline"  (MuZero Baseline),
                    "belief"    (Belief-Aware MuZero)
        simulations: number of MCTS simulations (only used for baseline/belief)
        """
        super().__init__(player_id, player_name)
        self._base_url = base_url.rstrip("/")
        self._rng_state_b64 = _make_rng_state_b64()
        self._agent_main = _make_agent_config(agent_type, simulations)
        self._agent_flip = _make_flip_agent_config(agent_type, simulations)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        try:
            return self._select_via_api(observation, legal_actions)
        except Exception as exc:
            print(f"[ApiPlayer] falling back to random ({exc})")
            print(f"actions: {legal_actions}")
            return random.choice(legal_actions)

    # ------------------------------------------------------------------
    # Phase dispatching
    # ------------------------------------------------------------------

    def _select_via_api(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        state = self._build_state(observation)
        phase = observation.turn_phase

        if phase == TurnPhase.STARTING_FLIPS:
            return self._handle_setup_reveal(state, observation)

        if phase == TurnPhase.CHOOSE_DRAW:
            return self._handle_choose_source(state, observation)

        if phase == TurnPhase.HAVE_DRAWN_HIDDEN:
            return self._handle_keep_or_discard(state, observation, legal_actions)

        if phase == TurnPhase.HAVE_DRAWN_OPEN:
            return self._handle_choose_position(
                state,
                observation,
                source="DISCARD",
                drawn_value=(
                    observation.hand_card.value if observation.hand_card else None
                ),
                keep=True,
            )

        if phase == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD:
            # The drawn card was just discarded, so it's now the discard top.
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
        context = self._make_context(observation.player_id, _PHASE_SETUP_REVEAL)
        resp = self._infer_step(state, context, agent=self._agent_flip)
        row, col = _api_pos_to_local(self._parse_pos(resp["step_log"]))
        return Action(ActionType.FLIP_CARD, pos=(row, col))

    def _handle_choose_source(self, state: dict, observation: Observation) -> Action:
        context = self._make_context(observation.player_id, _PHASE_CHOOSE_SOURCE)
        resp = self._infer_step(state, context)
        if "discard" in resp["step_log"]:
            return Action(ActionType.DRAW_OPEN_CARD)
        return Action(ActionType.DRAW_HIDDEN_CARD)

    def _handle_keep_or_discard(
        self, state: dict, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        drawn_value = observation.hand_card.value if observation.hand_card else None
        context = self._make_context(
            observation.player_id,
            _PHASE_KEEP_OR_DISCARD,
            source="DECK",
            drawn_value=drawn_value,
        )
        resp = self._infer_step(state, context)

        if "keep" in resp["step_log"]:
            # The response contains the context needed for the position step.
            next_ctx = resp.get("decision_context")
            if next_ctx is None:
                # Unexpected: fall back to first legal SWAP_CARD.
                for a in legal_actions:
                    if a.type == ActionType.SWAP_CARD:
                        return a
                return legal_actions[0]
            resp2 = self._infer_step(state, next_ctx)
            row, col = _api_pos_to_local(self._parse_pos(resp2["step_log"]))
            return Action(ActionType.SWAP_CARD, pos=(row, col))

        # Discard: find the DISCARD_CARD action.
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
        context = self._make_context(
            observation.player_id,
            _PHASE_CHOOSE_POSITION,
            source=source,
            drawn_value=drawn_value,
            keep=keep,
        )
        agent = self._agent_main if keep else self._agent_flip
        resp = self._infer_step(state, context, agent=agent)
        pos = self._parse_pos(resp["step_log"])
        row, col = _api_pos_to_local(pos)
        action_type = ActionType.SWAP_CARD if keep else ActionType.FLIP_CARD
        return Action(action_type, pos=(row, col))

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _infer_step(self, state: dict, decision_context, agent: dict = None) -> dict:
        payload = {
            "state": state,
            "agent": agent if agent is not None else self._agent_main,
            "decision_context": decision_context,
        }
        return _post_json(f"{self._base_url}/api/session/infer-agent-step", payload)

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_state(self, observation: Observation) -> dict:
        # opponent_cards is a list of length num_players, indexed by player_id.
        # Entry i is the opponent's grid if visible, None if it's our own slot or unknown.
        num_players = len(observation.opponent_cards)

        empty_board = {
            "cards": [0] * 12,
            "visible": [False] * 12,
            "removed": [False] * 12,
        }
        boards = [dict(empty_board) for _ in range(num_players)]
        boards[observation.player_id] = _card_grid_to_api_board(observation.card_grid)
        for i, opp_grid in enumerate(observation.opponent_cards):
            if i != observation.player_id and opp_grid is not None:
                boards[i] = _card_grid_to_api_board(opp_grid)

        phase = (
            "SETUP" if observation.turn_phase == TurnPhase.STARTING_FLIPS else "MAIN"
        )
        setup_reveals = [0] * num_players
        if observation.turn_phase == TurnPhase.STARTING_FLIPS:
            setup_reveals[observation.player_id] = 1

        discard_pile = []
        if observation.discard_top is not None:
            discard_pile = [observation.discard_top.value]

        return {
            "initial_seed": 0,
            "rng_state_b64": self._rng_state_b64,
            "num_players": num_players,
            "history_window_k": 16,
            "score_limit": 100,
            "setup_mode": "auto",
            "manual_initial_reveals": False,
            "round_index": 1,
            "global_step": 1,
            "turns_in_round": 1,
            "phase": phase,
            "current_player": observation.player_id,
            "scores": list(observation.scores),
            "round_scores": list(observation.scores),
            "deck": [0] * observation.draw_pile_size,
            "discard_pile": discard_pile,
            "boards": boards,
            "setup_reveals_remaining": setup_reveals,
            "pending_final_turn_players": [],
            "round_ender": None,
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

    @staticmethod
    def _parse_pos(step_log: str) -> int:
        """Extract the trailing integer position from a step_log string."""
        match = re.search(r"(\d+)\s*$", step_log)
        if match:
            return int(match.group(1))
        raise ValueError(f"No position found in step_log: {step_log!r}")
