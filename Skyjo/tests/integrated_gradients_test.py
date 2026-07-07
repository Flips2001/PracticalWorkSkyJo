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
    INITIAL_CARD_COUNTS,
    OBS_SIZE,
    encode_observation,
    normalize_card_value,
)
from Skyjo.src.rl.integrated_gradients import (
    FEATURE_METADATA,
    build_expected_card_baseline,
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


def test_feature_metadata_matches_encoding_size():
    assert len(FEATURE_METADATA) == OBS_SIZE
    assert [feature.index for feature in FEATURE_METADATA] == list(range(OBS_SIZE))


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


def test_expected_card_baseline_uses_remaining_draw_pile_counts():
    counts = [0] * len(CARD_VALUES)
    counts[CARD_VALUES.index(12)] = 10
    observation = _observation(draw_pile_value_counts=counts)
    observation.card_grid = [
        [Card(12, face_up=True), Card(0, face_up=False), Card(-2, face_up=True)],
        [Card(0, face_up=False), Card(0, face_up=False), Card(0, face_up=False)],
        [Card(0, face_up=False), Card(0, face_up=False), Card(0, face_up=False)],
    ]
    observation.hand_card = Card(-1, face_up=True)

    baseline = build_expected_card_baseline(observation)

    assert baseline[0] == pytest.approx(normalize_card_value(12))
    assert baseline[1] == pytest.approx(1.0)
    assert baseline[2] == pytest.approx(0.0)
    assert baseline[3] == pytest.approx(0.0)
    assert baseline[4] == pytest.approx(normalize_card_value(12))
    assert baseline[5] == pytest.approx(1.0)
    assert baseline[6] == pytest.approx(normalize_card_value(0))
    assert baseline[7] == pytest.approx(1.0)
    assert baseline[48] == pytest.approx(normalize_card_value(12))
    assert baseline[49] == pytest.approx(1.0)
    assert baseline[50] == pytest.approx(normalize_card_value(12))
    assert baseline[51] == pytest.approx(1.0)
    # Phase (CHOOSE_DRAW → index 53) is situational, so the baseline mirrors it.
    assert baseline[53] == pytest.approx(1.0)


def test_expected_card_baseline_holds_situational_features():
    counts = [0] * len(CARD_VALUES)
    counts[CARD_VALUES.index(12)] = 10
    observation = _observation(draw_pile_value_counts=counts)
    observation.card_grid = [
        [Card(7, face_up=True), Card(3, face_up=True), Card(0, face_up=False)],
        [Card(7, face_up=True), Card(4, face_up=True), Card(0, face_up=False)],
        [Card(0, face_up=False), Card(0, face_up=False), Card(0, face_up=False)],
    ]
    observation.scores = [15, 9]

    encoded = encode_observation(observation)
    baseline = build_expected_card_baseline(observation)

    # Situational features are copied from the observation (delta = 0).
    for index in range(52, 60):  # phase one-hot, scores, draw-pile size
        assert baseline[index] == pytest.approx(encoded[index])
    for index in range(60, OBS_SIZE):  # round-state flags, draw-pile counts
        assert baseline[index] == pytest.approx(encoded[index])


def test_expected_card_baseline_falls_back_to_initial_deck_expectation():
    observation = _observation(draw_pile_value_counts=[0] * len(CARD_VALUES))

    baseline = build_expected_card_baseline(observation)

    total_cards = sum(INITIAL_CARD_COUNTS.values())
    expected_card_value = (
        sum(value * count for value, count in INITIAL_CARD_COUNTS.items()) / total_cards
    )
    assert baseline[48] == pytest.approx(normalize_card_value(expected_card_value))


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


def test_explain_action_returns_ranked_features():
    model = DummyModel()
    action = Action(ActionType.DRAW_OPEN_CARD)
    legal_actions = [
        Action(ActionType.DRAW_HIDDEN_CARD),
        action,
    ]
    counts = [0] * len(CARD_VALUES)
    counts[CARD_VALUES.index(-2)] = 10

    explanation = explain_action(
        model,
        _observation(draw_pile_value_counts=counts),
        action,
        legal_actions,
        steps=32,
        top_k=2,
    )
    top_labels = [item.feature.label for item in explanation.top_features]

    assert explanation.error is None
    # Card features are attributed; the situational phase is fixed in the baseline.
    assert explanation.top_features[0].feature.label == "discard top value"
    assert "phase choose draw" not in top_labels
    assert explanation.action == action
    assert explanation.action_index == 1
    assert encode_observation(_observation()).shape == (OBS_SIZE,)


def test_summary_lines_show_card_influences_not_raw_encoded_values():
    model = DummyModel()
    action = Action(ActionType.DRAW_OPEN_CARD)
    counts = [0] * len(CARD_VALUES)
    counts[CARD_VALUES.index(-2)] = 10
    observation = _observation(draw_pile_value_counts=counts)
    observation.card_grid[0][0] = Card(12, face_up=True)

    explanation = explain_action(
        model,
        observation,
        action,
        [Action(ActionType.DRAW_HIDDEN_CARD), action],
        steps=32,
    )

    lines = explanation.summary_lines(max_features=3, include_action=False)

    assert any("own R0C0 card" in line for line in lines)
    assert any("discard top value" in line for line in lines)
    assert all("(value " not in line for line in lines)
    assert all(
        "toward " in line or "against " in line or "Total influence" in line
        for line in lines
    )
