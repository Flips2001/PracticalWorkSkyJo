import numpy as np
import pytest

from Skyjo.src.card import Card
from Skyjo.src.observation import Observation
from Skyjo.src.turn_phase import TurnPhase
from Skyjo.src.rl.encoding import (
    OBS_SIZE,
    encode_observation,
    get_observation_space,
    normalize_card_value,
)


def _make_grid(values, face_up=True):
    """Create a 3x4 grid of Cards from a flat list of 12 values."""
    grid = []
    for r in range(3):
        row = []
        for c in range(4):
            val = values[r * 4 + c]
            row.append(Card(value=val, face_up=face_up))
        grid.append(row)
    return grid


def _make_obs(**kwargs):
    """Create a minimal Observation with sensible defaults."""
    defaults = dict(
        player_id=0,
        card_grid=_make_grid([0] * 12, face_up=False),
        hand_card=None,
        opponent_cards=[_make_grid([0] * 12, face_up=False)],
        scores=[0, 0],
        discard_top=None,
        draw_pile_size=100,
        turn_phase=TurnPhase.CHOOSE_DRAW,
        draw_pile_value_counts=None,
    )
    defaults.update(kwargs)
    return Observation(**defaults)


class TestNormalizeCardValue:
    def test_min_value(self):
        assert normalize_card_value(-2) == pytest.approx(0.0)

    def test_max_value(self):
        assert normalize_card_value(12) == pytest.approx(1.0)

    def test_zero(self):
        assert normalize_card_value(0) == pytest.approx(2.0 / 14.0)


class TestEncodeObservation:
    def test_output_shape(self):
        obs = _make_obs()
        vec = encode_observation(obs)
        assert vec.shape == (OBS_SIZE,)

    def test_output_dtype(self):
        obs = _make_obs()
        vec = encode_observation(obs)
        assert vec.dtype == np.float32

    def test_own_grid_revealed_cards(self):
        grid = _make_grid([5] * 12, face_up=True)
        obs = _make_obs(card_grid=grid)
        vec = encode_observation(obs)
        # slot 0: value, slot 1: is_revealed
        assert vec[0] == pytest.approx(normalize_card_value(5))
        assert vec[1] == pytest.approx(1.0)

    def test_own_grid_hidden_cards(self):
        grid = _make_grid([5] * 12, face_up=False)
        obs = _make_obs(card_grid=grid)
        vec = encode_observation(obs)
        # hidden: value=0, is_revealed=0
        assert vec[0] == pytest.approx(0.0)
        assert vec[1] == pytest.approx(0.0)

    def test_discard_top_present(self):
        obs = _make_obs(discard_top=Card(value=7, face_up=True))
        vec = encode_observation(obs)
        assert vec[48] == pytest.approx(normalize_card_value(7))
        assert vec[49] == pytest.approx(1.0)

    def test_discard_top_absent(self):
        obs = _make_obs(discard_top=None)
        vec = encode_observation(obs)
        assert vec[48] == pytest.approx(0.0)
        assert vec[49] == pytest.approx(0.0)

    def test_hand_card_present(self):
        obs = _make_obs(hand_card=Card(value=3, face_up=True))
        vec = encode_observation(obs)
        assert vec[50] == pytest.approx(normalize_card_value(3))
        assert vec[51] == pytest.approx(1.0)

    def test_hand_card_absent(self):
        obs = _make_obs(hand_card=None)
        vec = encode_observation(obs)
        assert vec[50] == pytest.approx(0.0)
        assert vec[51] == pytest.approx(0.0)

    def test_turn_phase_one_hot(self):
        obs = _make_obs(turn_phase=TurnPhase.CHOOSE_DRAW)
        vec = encode_observation(obs)
        # CHOOSE_DRAW is index 1 in _PHASE_ORDER
        assert vec[52] == pytest.approx(0.0)  # STARTING_FLIPS
        assert vec[53] == pytest.approx(1.0)  # CHOOSE_DRAW
        assert vec[54] == pytest.approx(0.0)
        assert vec[55] == pytest.approx(0.0)
        assert vec[56] == pytest.approx(0.0)

    def test_turn_phase_starting_flips(self):
        obs = _make_obs(turn_phase=TurnPhase.STARTING_FLIPS)
        vec = encode_observation(obs)
        assert vec[52] == pytest.approx(1.0)

    def test_scores_are_not_encoded(self):
        # Round scores are derivable from revealed cards; the model must
        # learn them, so identical boards with different score fields encode
        # identically.
        vec_a = encode_observation(_make_obs(scores=[50, 75]))
        vec_b = encode_observation(_make_obs(scores=[0, 0]))
        assert np.array_equal(vec_a, vec_b)

    def test_draw_pile_size_normalized(self):
        obs = _make_obs(draw_pile_size=75)
        vec = encode_observation(obs)
        assert vec[57] == pytest.approx(75.0 / 150.0)

    def test_final_turn_flag(self):
        obs = _make_obs(final_turn_phase=True)
        vec = encode_observation(obs)
        assert vec[58] == pytest.approx(1.0)

    def test_not_final_turn(self):
        obs = _make_obs(final_turn_phase=False)
        vec = encode_observation(obs)
        assert vec[58] == pytest.approx(0.0)

    def test_first_finisher_flag(self):
        obs = _make_obs(first_finisher_id=0)  # player_id=0 is finisher
        vec = encode_observation(obs)
        assert vec[59] == pytest.approx(1.0)

    def test_not_first_finisher(self):
        obs = _make_obs(first_finisher_id=1)  # opponent is finisher
        vec = encode_observation(obs)
        assert vec[59] == pytest.approx(0.0)

    def test_draw_pile_value_counts_absent(self):
        obs = _make_obs(draw_pile_value_counts=None)
        vec = encode_observation(obs)
        assert np.allclose(vec[60:75], 0.0)

    def test_draw_pile_value_counts_encoded(self):
        counts = [0] * 15
        counts[0] = 5  # value -2 (max 5)
        counts[2] = 9  # value 0 (max 15)
        counts[14] = 5  # value 12 (max 10)
        obs = _make_obs(draw_pile_value_counts=counts)
        vec = encode_observation(obs)

        assert vec[60] == pytest.approx(1.0)
        assert vec[62] == pytest.approx(9.0 / 15.0)
        assert vec[74] == pytest.approx(0.5)


class TestObservationSpace:
    def test_shape(self):
        space = get_observation_space()
        assert space.shape == (OBS_SIZE,)

    def test_dtype(self):
        space = get_observation_space()
        assert space.dtype == np.float32

    def test_encoded_obs_in_space(self):
        obs = _make_obs()
        vec = encode_observation(obs)
        space = get_observation_space()
        assert space.contains(vec)
