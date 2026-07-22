import numpy as np
import pytest
import torch

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.observation import Observation
from Skyjo.src.rl.action_mapping import NUM_ACTIONS
from Skyjo.src.rl.encoding import (
    CARD_VALUES,
    OBS_SIZE,
    encode_observation,
    expected_card_value,
    normalize_card_value,
)
from Skyjo.src.rl.integrated_gradients import (
    build_blindfold_baseline,
    explain_action,
    integrated_gradients,
)
from Skyjo.src.turn_phase import TurnPhase


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(OBS_SIZE, NUM_ACTIONS, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[1, 0] = 1.5
            self.linear.weight[1, 48] = 2.0
            self.linear.weight[1, 53] = 1.0

    @property
    def device(self):
        return self.linear.weight.device

    def get_distribution(self, obs, action_masks=None):
        logits = self.linear(obs)
        if action_masks is not None:
            logits = torch.where(action_masks, logits, torch.full_like(logits, -1e8))
        return torch.distributions.Categorical(logits=logits)


class DummyModel:
    def __init__(self):
        self.policy = DummyPolicy()


def _hidden_grid(cols=4):
    return [[Card(0, face_up=False) for _ in range(cols)] for _ in range(3)]


def _observation(draw_pile_value_counts=None):
    return Observation(
        player_id=0,
        card_grid=_hidden_grid(),
        hand_card=None,
        opponent_cards=[None, _hidden_grid()],
        scores=[0, 0],
        discard_top=Card(5, face_up=True),
        draw_pile_size=100,
        turn_phase=TurnPhase.CHOOSE_DRAW,
        draw_pile_value_counts=draw_pile_value_counts or [0] * 15,
    )


def test_integrated_gradients_matches_chosen_action_log_prob_delta():
    model = DummyModel()
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[48] = 0.5
    obs[53] = 1.0
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[0] = 1
    mask[1] = 1

    attributions, target_score, baseline_score = integrated_gradients(
        model,
        obs,
        action_index=1,
        action_mask=mask,
        steps=128,
    )

    assert attributions[48] > 0
    assert attributions[53] > 0
    assert attributions.sum() == pytest.approx(target_score - baseline_score, abs=1e-4)


def test_blindfold_baseline_erases_card_knowledge():
    observation = _observation()
    observation.card_grid[0][0] = Card(12, face_up=True)
    observation.hand_card = Card(-1, face_up=True)

    baseline = build_blindfold_baseline(observation)
    expected = normalize_card_value(
        expected_card_value(observation.draw_pile_value_counts)
    )

    # Grids hidden.
    assert baseline[:48] == pytest.approx(np.zeros(48))
    # Discard and hand exist publicly: presence kept, value erased to the
    # expected remaining-deck card.
    assert baseline[48] == pytest.approx(expected)
    assert baseline[49] == pytest.approx(1.0)
    assert baseline[50] == pytest.approx(expected)
    assert baseline[51] == pytest.approx(1.0)
    # Deck counts revert to the full deck: nothing seen yet.
    assert baseline[60:] == pytest.approx(np.ones(len(CARD_VALUES)))


def test_blindfold_baseline_keeps_absent_hand_absent():
    observation = _observation()  # no hand card

    baseline = build_blindfold_baseline(observation)

    assert baseline[50] == pytest.approx(0.0)
    assert baseline[51] == pytest.approx(0.0)


def test_blindfold_baseline_keeps_public_context():
    observation = _observation()
    observation.card_grid = [row[:3] for row in _hidden_grid()]  # column 3 removed
    encoded = encode_observation(observation)

    baseline = build_blindfold_baseline(observation)

    # Phase one-hot, draw-pile size, round flags mirror the observation.
    for index in range(52, 60):
        assert baseline[index] == pytest.approx(encoded[index])
    # Removed slots are public structure: copied as (normalize(0), revealed).
    for row in range(3):
        index = (row * 4 + 3) * 2
        assert baseline[index] == pytest.approx(normalize_card_value(0))
        assert baseline[index + 1] == pytest.approx(1.0)


def test_integrated_gradients_rejects_masked_selected_action():
    model = DummyModel()
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[0] = 1

    with pytest.raises(ValueError, match="masked as illegal"):
        integrated_gradients(
            model,
            obs,
            action_index=1,
            action_mask=mask,
            steps=8,
        )


def test_integrated_gradients_restores_policy_training_mode():
    model = DummyModel()
    model.policy.train()
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)

    integrated_gradients(
        model,
        obs,
        action_index=1,
        action_mask=mask,
        steps=8,
    )

    assert model.policy.training is True


def test_explain_action_attributes_card_units_not_context():
    model = DummyModel()
    action = Action(ActionType.DRAW_OPEN_CARD)
    observation = _observation()
    observation.card_grid[0][0] = Card(12, face_up=True)
    # Well above the expected-value baseline so the discard delta is positive.
    observation.discard_top = Card(12, face_up=True)

    explanation = explain_action(
        model,
        observation,
        action,
        [Action(ActionType.DRAW_HIDDEN_CARD), action],
        steps=32,
    )

    assert explanation.error is None
    assert explanation.action == action
    assert explanation.action_index == 1

    by_label = {unit.label: unit for unit in explanation.units}
    # Weighted card inputs (own R0C0 value, discard value) carry attribution.
    assert by_label["your 12 at R0C0"].attribution > 0
    assert by_label["discard 12"].attribution > 0
    # The phase weight is public context and produces no unit at all.
    assert not any("phase" in label for label in by_label)
    # Hidden cards have zero delta and therefore exactly zero attribution.
    assert by_label["your hidden card at R1C1"].attribution == 0.0
    # Deck counts differ from the full-deck baseline and get attributed.
    assert "remaining -2s in deck" in by_label


def test_explanation_lookup_helpers_for_ui():
    model = DummyModel()
    action = Action(ActionType.DRAW_OPEN_CARD)
    observation = _observation()
    observation.card_grid[0][0] = Card(12, face_up=True)

    explanation = explain_action(
        model,
        observation,
        action,
        [Action(ActionType.DRAW_HIDDEN_CARD), action],
    )

    assert explanation.max_abs_attribution > 0
    assert explanation.unit_for("discard").label == "discard 5"
    assert explanation.unit_for("hand") is None
    assert set(explanation.deck_map()) == set(CARD_VALUES)


def test_grid_map_covers_all_cells_for_colouring():
    model = DummyModel()
    action = Action(ActionType.DRAW_OPEN_CARD)
    observation = _observation()
    observation.card_grid[0][0] = Card(12, face_up=True)

    explanation = explain_action(
        model,
        observation,
        action,
        [Action(ActionType.DRAW_HIDDEN_CARD), action],
    )

    own = explanation.grid_map("own")
    assert set(own) == {(r, c) for r in range(3) for c in range(4)}
    assert own[(0, 0)].abs_attribution > 0
    assert len(explanation.grid_map("opponent")) == 12


def test_total_influence_is_zero_for_a_card_blind_policy():
    model = DummyModel()
    with torch.no_grad():
        model.policy.linear.weight.zero_()
    action = Action(ActionType.DRAW_OPEN_CARD)

    explanation = explain_action(
        model,
        _observation(),
        action,
        [Action(ActionType.DRAW_HIDDEN_CARD), action],
    )

    assert explanation.total_influence == pytest.approx(0.0)
    assert explanation.max_abs_attribution == pytest.approx(0.0)
