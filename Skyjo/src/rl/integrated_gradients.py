"""Integrated-gradients explanations for RL moves.

The baseline is the same public situation with all card knowledge erased:
cards hidden, discard/hand absent, deck counts at "nothing seen", revealed
totals zero. Phase, draw-pile size, round flags and removed columns are
public regardless of card knowledge, so they are copied from the observation
and cancel. Attributions therefore answer: how much did each piece of card
knowledge push the policy toward the chosen move, relative to an agent in
the identical situation that cannot see any cards.

Value and revealed-flag features are summed per card: normalize(-2) == 0.0,
so against the hidden baseline the split between the two channels is an
encoding artifact and only their sum is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.rl.action_mapping import NUM_ACTIONS, action_to_int, legal_actions_mask
from Skyjo.src.rl.encoding import (
    CARD_VALUES,
    GRID_COLS,
    GRID_ROWS,
    OBS_SIZE,
    encode_observation,
    normalize_card_value,
)


GridPos = Tuple[int, int]

# Encoding layout (see encoding.py). Grid slots are (value, revealed) pairs.
_OWN_GRID_OFFSET = 0
_OPPONENT_GRID_OFFSET = 24
_DISCARD_INDEX = 48
_HAND_INDEX = 50
_OWN_SCORE_INDEX = 57
_OPPONENT_SCORE_INDEX = 58
_DECK_COUNTS_OFFSET = 62

# Public context regardless of card knowledge: phase one-hot, draw-pile size,
# final-turn and first-finisher flags.
_PUBLIC_CONTEXT_INDICES = tuple(range(52, 57)) + (59, 60, 61)

# Below this log-prob delta vs the blindfold baseline (and top unit magnitude),
# card knowledge did not drive the move and ranked influences would be noise.
LOW_INFLUENCE_THRESHOLD = 0.25


@dataclass(frozen=True)
class UnitAttribution:
    """Signed influence of one piece of card knowledge on the chosen move."""

    label: str
    attribution: float
    owner: Optional[str] = None
    pos: Optional[GridPos] = None

    @property
    def abs_attribution(self) -> float:
        return abs(self.attribution)


@dataclass(frozen=True)
class ActionExplanation:
    """Explanation bundle for one selected action."""

    action: Action
    action_index: int
    units: List[UnitAttribution] = field(default_factory=list)
    target_score: Optional[float] = None
    baseline_score: Optional[float] = None
    error: Optional[str] = None

    @property
    def total_influence(self) -> Optional[float]:
        if self.target_score is None or self.baseline_score is None:
            return None
        return self.target_score - self.baseline_score

    def summary_lines(
        self, max_features: int = 5, include_action: bool = True
    ) -> List[str]:
        if self.error:
            return [f"Attribution unavailable: {self.error}"]

        lines = [f"RL chose: {self.action}"] if include_action else []
        ranked = sorted(
            (unit for unit in self.units if unit.abs_attribution > 0),
            key=lambda unit: unit.abs_attribution,
            reverse=True,
        )

        total = self.total_influence
        if total is not None:
            lines.append(f"Card knowledge influence: {total:+.3f}")
            top = ranked[0].abs_attribution if ranked else 0.0
            if abs(total) < LOW_INFLUENCE_THRESHOLD and top < LOW_INFLUENCE_THRESHOLD:
                lines.append("Card knowledge had little influence on this move.")
                return lines

        for unit in ranked[:max_features]:
            direction = "toward" if unit.attribution >= 0 else "against"
            lines.append(f"{direction} {unit.label}: {unit.attribution:+.3f}")

        return lines

    def grid_map(self, owner: str) -> Dict[GridPos, UnitAttribution]:
        return {
            unit.pos: unit
            for unit in self.units
            if unit.owner == owner and unit.pos is not None
        }


def build_blindfold_baseline(observation: Observation) -> np.ndarray:
    """Erase all card knowledge, keep the public situation."""
    baseline = np.zeros(OBS_SIZE, dtype=np.float32)
    baseline[_DECK_COUNTS_OFFSET:] = 1.0  # full deck: nothing seen yet

    encoded = encode_observation(observation)
    for index in _PUBLIC_CONTEXT_INDICES:
        baseline[index] = encoded[index]

    _copy_removed_slots(baseline, observation.card_grid, _OWN_GRID_OFFSET)
    _copy_removed_slots(baseline, _opponent_grid(observation), _OPPONENT_GRID_OFFSET)
    return baseline


def integrated_gradients(
    model,
    observation_vector: np.ndarray,
    action_index: int,
    action_mask: Optional[np.ndarray] = None,
    *,
    steps: int = 32,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Compute integrated gradients for the selected action log-probability.

    Returns:
        Tuple of (attributions, target_log_prob, baseline_log_prob).
    """
    obs = np.asarray(observation_vector, dtype=np.float32)
    if obs.shape != (OBS_SIZE,):
        raise ValueError(f"Expected observation shape {(OBS_SIZE,)}, got {obs.shape}.")

    base = (
        np.zeros_like(obs)
        if baseline is None
        else np.asarray(baseline, dtype=np.float32)
    )
    if base.shape != obs.shape:
        raise ValueError(f"Expected baseline shape {obs.shape}, got {base.shape}.")

    if action_index < 0 or action_index >= NUM_ACTIONS:
        raise ValueError(f"action_index must be in [0, {NUM_ACTIONS}).")

    steps = max(1, int(steps))
    policy = model.policy
    device = torch.device(getattr(policy, "device", getattr(model, "device", "cpu")))
    was_training = bool(getattr(policy, "training", False))
    policy.eval()

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    base_tensor = torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0)
    delta = obs_tensor - base_tensor

    masks = _prepare_action_masks(action_mask, steps, device, action_index)
    single_mask = masks[:1] if masks is not None else None

    try:
        alphas = (
            (torch.arange(steps, dtype=torch.float32, device=device) + 0.5) / steps
        ).unsqueeze(1)
        scaled_inputs = base_tensor + alphas * delta
        scaled_inputs.requires_grad_(True)

        log_probs = _target_action_log_probs(policy, scaled_inputs, action_index, masks)
        grads = torch.autograd.grad(log_probs.sum(), scaled_inputs)[0]
        attributions = (delta.squeeze(0) * grads.mean(dim=0)).detach().cpu().numpy()

        with torch.no_grad():
            target_score = float(
                _target_action_log_probs(
                    policy, obs_tensor, action_index, single_mask
                ).item()
            )
            baseline_score = float(
                _target_action_log_probs(
                    policy, base_tensor, action_index, single_mask
                ).item()
            )
    finally:
        policy.train(was_training)

    return attributions.astype(np.float32), target_score, baseline_score


def explain_action(
    model,
    observation: Observation,
    action: Action,
    legal_actions: Iterable[Action],
    *,
    steps: int = 32,
) -> ActionExplanation:
    """Build an integrated-gradients explanation for a selected action."""
    action_index = action_to_int(action)

    attributions, target_score, baseline_score = integrated_gradients(
        model=model,
        observation_vector=encode_observation(observation),
        action_index=action_index,
        action_mask=legal_actions_mask(list(legal_actions)),
        steps=steps,
        baseline=build_blindfold_baseline(observation),
    )

    return ActionExplanation(
        action=action,
        action_index=action_index,
        units=_build_units(observation, attributions),
        target_score=target_score,
        baseline_score=baseline_score,
    )


def unavailable_explanation(action: Action, error: Exception) -> ActionExplanation:
    return ActionExplanation(
        action=action,
        action_index=action_to_int(action),
        error=str(error),
    )


def _opponent_grid(observation: Observation):
    return next((g for g in observation.opponent_cards if g is not None), None)


def _slot_is_removed(grid, row: int, col: int) -> bool:
    # Mirrors _encode_grid: a slot outside the grid is a removed column.
    return grid is None or row >= len(grid) or col >= len(grid[row])


def _copy_removed_slots(baseline: np.ndarray, grid, offset: int) -> None:
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if _slot_is_removed(grid, row, col):
                index = offset + (row * GRID_COLS + col) * 2
                baseline[index] = normalize_card_value(0)
                baseline[index + 1] = 1.0


def _slot_label(prefix: str, grid, row: int, col: int) -> str:
    if _slot_is_removed(grid, row, col):
        return f"{prefix} removed slot at R{row}C{col}"
    card = grid[row][col]
    if card is not None and card.face_up:
        return f"{prefix} {card.value} at R{row}C{col}"
    return f"{prefix} hidden card at R{row}C{col}"


def _build_units(
    observation: Observation, attributions: np.ndarray
) -> List[UnitAttribution]:
    units: List[UnitAttribution] = []

    for owner, prefix, grid, offset in (
        ("own", "your", observation.card_grid, _OWN_GRID_OFFSET),
        ("opponent", "opponent's", _opponent_grid(observation), _OPPONENT_GRID_OFFSET),
    ):
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                index = offset + (row * GRID_COLS + col) * 2
                units.append(
                    UnitAttribution(
                        label=_slot_label(prefix, grid, row, col),
                        attribution=float(
                            attributions[index] + attributions[index + 1]
                        ),
                        owner=owner,
                        pos=(row, col),
                    )
                )

    for label_prefix, index, card in (
        ("discard", _DISCARD_INDEX, observation.discard_top),
        ("hand card", _HAND_INDEX, observation.hand_card),
    ):
        if card is not None:
            units.append(
                UnitAttribution(
                    label=f"{label_prefix} {card.value}",
                    attribution=float(attributions[index] + attributions[index + 1]),
                )
            )

    units.append(
        UnitAttribution(
            label="your revealed total",
            attribution=float(attributions[_OWN_SCORE_INDEX]),
        )
    )
    units.append(
        UnitAttribution(
            label="opponent revealed total",
            attribution=float(attributions[_OPPONENT_SCORE_INDEX]),
        )
    )

    for offset, value in enumerate(CARD_VALUES):
        units.append(
            UnitAttribution(
                label=f"remaining {value}s in deck",
                attribution=float(attributions[_DECK_COUNTS_OFFSET + offset]),
            )
        )

    return units


def _prepare_action_masks(
    action_mask: Optional[np.ndarray],
    batch_size: int,
    device: torch.device,
    action_index: int,
) -> Optional[torch.Tensor]:
    if action_mask is None:
        return None
    mask = np.asarray(action_mask, dtype=np.bool_)
    if mask.shape != (NUM_ACTIONS,):
        raise ValueError(
            f"Expected action mask shape {(NUM_ACTIONS,)}, got {mask.shape}."
        )
    if not bool(mask[action_index]):
        raise ValueError(f"action_index {action_index} is masked as illegal.")
    return torch.as_tensor(mask, dtype=torch.bool, device=device).expand(batch_size, -1)


def _target_action_log_probs(
    policy,
    obs_tensor: torch.Tensor,
    action_index: int,
    action_masks: Optional[torch.Tensor],
) -> torch.Tensor:
    distribution = policy.get_distribution(obs_tensor, action_masks=action_masks)
    actions = torch.full(
        (obs_tensor.shape[0],),
        action_index,
        dtype=torch.long,
        device=obs_tensor.device,
    )
    return distribution.log_prob(actions)
