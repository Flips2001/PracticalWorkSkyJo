import math

import pytest

from Skyjo.src.rl import train


def test_mean_ci95_empty_returns_zero():
    assert train._mean_ci95([]) == (0.0, 0.0)


def test_mean_ci95_single_has_no_interval():
    assert train._mean_ci95([5.0]) == (5.0, 0.0)


def test_mean_ci95_constant_values_have_zero_width():
    mean, ci = train._mean_ci95([10, 10, 10, 10])
    assert mean == pytest.approx(10.0)
    assert ci == pytest.approx(0.0)


def test_mean_ci95_matches_formula():
    values = [1, 2, 3, 4, 5]
    mean, ci = train._mean_ci95(values)
    std = math.sqrt(2.5)  # sample std (ddof=1) of 1..5
    assert mean == pytest.approx(3.0)
    assert ci == pytest.approx(1.96 * std / math.sqrt(len(values)))


def test_evaluate_matchup_aggregates_and_alternates_seats(monkeypatch):
    seats = []

    def _fake_game(model, make_opponent, rl_seat, rl_deterministic=True):
        seats.append(rl_seat)
        # RL scores 10, opponent 20 -> RL wins by margin +10, one clear.
        return 10, 20, 1

    monkeypatch.setattr(train, "_play_matchup_game", _fake_game)

    result = train.evaluate_matchup(
        model=None, make_opponent=lambda pid: None, num_games=4
    )

    assert seats == [0, 1, 0, 1]  # seats alternate across games
    assert result["winrate"] == pytest.approx(100.0)
    assert result["rl_avg"] == pytest.approx(10.0)
    assert result["opp_avg"] == pytest.approx(20.0)
    assert result["margin"] == pytest.approx(10.0)
    assert result["margin_ci"] == pytest.approx(0.0)
    assert result["clears"] == pytest.approx(1.0)


def test_evaluate_matchup_counts_only_strict_wins(monkeypatch):
    # RL and opponent tie every game -> no wins, zero margin.
    monkeypatch.setattr(train, "_play_matchup_game", lambda *a, **k: (15, 15, 0))

    result = train.evaluate_matchup(
        model=None, make_opponent=lambda pid: None, num_games=3
    )

    assert result["winrate"] == pytest.approx(0.0)
    assert result["margin"] == pytest.approx(0.0)


def test_phillips_opponent_uses_deterministic_rl_side():
    opponent = train.phillips_opponent()
    assert opponent.name == "Phillips"
    assert opponent.rl_deterministic is True


def test_frozen_mirror_opponent_samples_both_sides():
    # The mirror must sample actions (deterministic=False) on BOTH sides, or two
    # greedy equal policies can lock into non-terminating rounds.
    opponent = train.frozen_mirror_opponent(reference_model=object())

    assert opponent.rl_deterministic is False
    reference_player = opponent.make_opponent(1)
    assert reference_player.deterministic is False


def test_evaluate_opponent_forwards_rl_determinism(monkeypatch):
    captured = {}

    def _fake_game(model, make_opponent, rl_seat, rl_deterministic=True):
        captured["rl_deterministic"] = rl_deterministic
        return 10, 20, 0

    monkeypatch.setattr(train, "_play_matchup_game", _fake_game)
    train.evaluate_opponent(
        object(), train.frozen_mirror_opponent(object()), num_games=1
    )

    assert captured["rl_deterministic"] is False


def test_selection_key_prefers_winrate_then_margin():
    # Win rate dominates; the margin only breaks ties.
    assert train._selection_key({"winrate": 90, "margin": 5.0}) < train._selection_key(
        {"winrate": 91, "margin": -100.0}
    )
    assert train._selection_key({"winrate": 90, "margin": 5.0}) < train._selection_key(
        {"winrate": 90, "margin": 6.0}
    )
