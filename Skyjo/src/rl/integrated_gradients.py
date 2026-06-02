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
    expected_card_value,
    normalize_card_value,
)


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class EncodedFeature:
    """Metadata for one scalar in the 85-dimensional observation vector."""

    index: int
    label: str
    group: str
    owner: Optional[str] = None
    pos: Optional[GridPos] = None


@dataclass(frozen=True)
class FeatureAttribution:
    """Integrated-gradient attribution for a single encoded feature."""

    feature: EncodedFeature
    value: float
    attribution: float

    @property
    def abs_attribution(self) -> float:
        return abs(self.attribution)


@dataclass(frozen=True)
class CellAttribution:
    """Aggregated attribution for one visible grid cell."""

    owner: str
    pos: GridPos
    attribution: float
    abs_attribution: float


@dataclass(frozen=True)
class ActionExplanation:
    """Explanation bundle for one selected action."""

    action: Action
    action_index: int
    top_features: List[FeatureAttribution] = field(default_factory=list)
    cell_attributions: List[CellAttribution] = field(default_factory=list)
    target_score: Optional[float] = None
    baseline_score: Optional[float] = None
    error: Optional[str] = None

    def summary_lines(
        self, max_features: int = 5, include_action: bool = True
    ) -> List[str]:
        if self.error:
            return [f"Attribution unavailable: {self.error}"]

        lines = [f"RL chose: {self.action}"] if include_action else []
        if self.target_score is not None and self.baseline_score is not None:
            delta = self.target_score - self.baseline_score
            lines.append(f"Total influence vs baseline: {delta:+.3f}")

        items = _summary_items(self.cell_attributions, self.top_features)
        lines.extend(line for _, line in items[:max_features])

        return lines

    def grid_map(self, owner: str) -> Dict[GridPos, CellAttribution]:
        return {
            cell.pos: cell for cell in self.cell_attributions if cell.owner == owner
        }


def build_feature_metadata() -> List[EncodedFeature]:
    """Return metadata for every scalar produced by encode_observation()."""
    features: List[EncodedFeature] = []

    def add_grid(owner: str, offset: int):
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                base = offset + (row * GRID_COLS + col) * 2
                prefix = f"{owner} R{row}C{col}"
                features.append(
                    EncodedFeature(
                        index=base,
                        label=f"{prefix} value",
                        group=f"{owner}_grid",
                        owner=owner,
                        pos=(row, col),
                    )
                )
                features.append(
                    EncodedFeature(
                        index=base + 1,
                        label=f"{prefix} revealed",
                        group=f"{owner}_grid",
                        owner=owner,
                        pos=(row, col),
                    )
                )

    add_grid("own", 0)
    add_grid("opponent", 24)

    features.extend(
        [
            EncodedFeature(48, "discard top value", "discard"),
            EncodedFeature(49, "discard top present", "discard"),
            EncodedFeature(50, "hand card value", "hand"),
            EncodedFeature(51, "hand card present", "hand"),
            EncodedFeature(52, "phase starting flips", "phase"),
            EncodedFeature(53, "phase choose draw", "phase"),
            EncodedFeature(54, "phase drawn hidden", "phase"),
            EncodedFeature(55, "phase drawn open", "phase"),
            EncodedFeature(56, "phase must flip after discard", "phase"),
            EncodedFeature(57, "own round score", "score"),
            EncodedFeature(58, "opponent round score", "score"),
            EncodedFeature(59, "draw pile size", "draw_pile"),
        ]
    )

    for col in range(GRID_COLS):
        features.append(
            EncodedFeature(
                60 + col,
                f"own column {col} match count",
                "column_matches",
                owner="own",
            )
        )
    for col in range(GRID_COLS):
        features.append(
            EncodedFeature(
                64 + col,
                f"opponent column {col} match count",
                "column_matches",
                owner="opponent",
            )
        )

    features.extend(
        [
            EncodedFeature(68, "final turn flag", "round_state"),
            EncodedFeature(69, "is first finisher", "round_state"),
        ]
    )

    for offset, value in enumerate(CARD_VALUES):
        features.append(
            EncodedFeature(
                70 + offset,
                f"draw pile remaining {value}",
                "draw_pile_counts",
            )
        )

    features.sort(key=lambda item: item.index)
    if len(features) != OBS_SIZE or any(
        item.index != i for i, item in enumerate(features)
    ):
        raise RuntimeError(
            "Integrated-gradients feature metadata does not match encoding."
        )
    return features


FEATURE_METADATA = build_feature_metadata()


def _summary_items(
    cell_attributions: Iterable[CellAttribution],
    feature_attributions: Iterable[FeatureAttribution],
) -> List[Tuple[float, str]]:
    items: List[Tuple[float, str]] = []

    for cell in cell_attributions:
        if cell.abs_attribution <= 0:
            continue
        row, col = cell.pos
        direction = "toward" if cell.attribution >= 0 else "against"
        items.append(
            (
                cell.abs_attribution,
                f"{direction} {cell.owner} R{row}C{col} card: {cell.attribution:+.3f}",
            )
        )

    for item in feature_attributions:
        if item.feature.pos is not None or item.abs_attribution <= 0:
            continue
        direction = "toward" if item.attribution >= 0 else "against"
        items.append(
            (
                item.abs_attribution,
                f"{direction} {item.feature.label}: {item.attribution:+.3f}",
            )
        )

    return sorted(items, key=lambda item: item[0], reverse=True)


def build_expected_card_baseline(observation: Observation) -> np.ndarray:
    """Build a neutral baseline that preserves known card structure.

    Visible card values are replaced by the expected remaining card value.
    Hidden cards stay hidden, removed columns stay removed, and non-card features
    use the default zero baseline.
    """
    baseline = np.zeros(OBS_SIZE, dtype=np.float32)
    expected_value = normalize_card_value(
        expected_card_value(observation.draw_pile_value_counts)
    )

    _apply_grid_baseline(baseline, observation.card_grid, 0, expected_value)
    opponent_grid = next((g for g in observation.opponent_cards if g is not None), None)
    _apply_grid_baseline(baseline, opponent_grid, 24, expected_value)

    for value_index, present_index, card in (
        (48, 49, observation.discard_top),
        (50, 51, observation.hand_card),
    ):
        if card is not None:
            baseline[value_index] = expected_value
            baseline[present_index] = 1.0

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
    top_k: int = 8,
) -> ActionExplanation:
    """Build an integrated-gradients explanation for a selected action."""
    action_index = action_to_int(action)
    obs_vec = encode_observation(observation)
    mask = legal_actions_mask(list(legal_actions))

    attributions, target_score, baseline_score = integrated_gradients(
        model=model,
        observation_vector=obs_vec,
        action_index=action_index,
        action_mask=mask,
        steps=steps,
        baseline=build_expected_card_baseline(observation),
    )

    feature_attributions = [
        FeatureAttribution(
            feature=feature,
            value=float(obs_vec[feature.index]),
            attribution=float(attributions[feature.index]),
        )
        for feature in FEATURE_METADATA
    ]
    top_features = sorted(
        feature_attributions,
        key=lambda item: item.abs_attribution,
        reverse=True,
    )[:top_k]

    return ActionExplanation(
        action=action,
        action_index=action_index,
        top_features=top_features,
        cell_attributions=_aggregate_cell_attributions(feature_attributions),
        target_score=target_score,
        baseline_score=baseline_score,
    )


def unavailable_explanation(action: Action, error: Exception) -> ActionExplanation:
    return ActionExplanation(
        action=action,
        action_index=action_to_int(action),
        error=str(error),
    )


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


def _apply_grid_baseline(
    baseline: np.ndarray,
    grid,
    offset: int,
    baseline_card_value: float,
) -> None:
    removed_value = normalize_card_value(0)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            index = offset + (row * GRID_COLS + col) * 2
            if grid is not None and row < len(grid) and col < len(grid[row]):
                card = grid[row][col]
                if card is not None and card.face_up:
                    baseline[index] = baseline_card_value
                    baseline[index + 1] = 1.0
            else:
                baseline[index] = removed_value
                baseline[index + 1] = 1.0


def _aggregate_cell_attributions(
    feature_attributions: Iterable[FeatureAttribution],
) -> List[CellAttribution]:
    totals: Dict[Tuple[str, GridPos], float] = {}
    absolute_totals: Dict[Tuple[str, GridPos], float] = {}

    for item in feature_attributions:
        owner = item.feature.owner
        pos = item.feature.pos
        if owner not in {"own", "opponent"} or pos is None:
            continue

        key = (owner, pos)
        totals[key] = totals.get(key, 0.0) + item.attribution
        absolute_totals[key] = absolute_totals.get(key, 0.0) + item.abs_attribution

    return [
        CellAttribution(
            owner=owner,
            pos=pos,
            attribution=totals[(owner, pos)],
            abs_attribution=absolute_totals[(owner, pos)],
        )
        for owner, pos in sorted(totals)
    ]
