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


def test_selfplay_opponent_samples_both_sides():
    # The self-play opponent must sample actions (deterministic=False) on BOTH
    # sides, or two near-identical greedy policies can stall a round.
    opponent = train.selfplay_opponent(model=object())

    assert opponent.name == "selfplay"
    assert opponent.rl_deterministic is False
    selfplay_player = opponent.make_opponent(1)
    assert selfplay_player.deterministic is False


def test_evaluate_opponent_forwards_rl_determinism(monkeypatch):
    captured = {}

    def _fake_game(model, make_opponent, rl_seat, rl_deterministic=True):
        captured["rl_deterministic"] = rl_deterministic
        return 10, 20, 0

    monkeypatch.setattr(train, "_play_matchup_game", _fake_game)
    train.evaluate_opponent(object(), train.selfplay_opponent(object()), num_games=1)

    assert captured["rl_deterministic"] is False


def test_selection_key_prefers_winrate_then_margin():
    # Win rate dominates; the margin only breaks ties.
    assert train._selection_key({"winrate": 90, "margin": 5.0}) < train._selection_key(
        {"winrate": 91, "margin": -100.0}
    )
    assert train._selection_key({"winrate": 90, "margin": 5.0}) < train._selection_key(
        {"winrate": 90, "margin": 6.0}
    )


# --- Sparring-partner promotion --------------------------------------------
# The gate must read the self-play result and nothing else. Tying it to the
# primary opponent's win rate can deadlock: a weak partner drags that win rate
# down, and a falling win rate never clears its own high-water mark, so the
# partner stays frozen for the rest of the run.


class _FakeLogger:
    def record(self, *args, **kwargs):
        pass

    def dump(self, *args, **kwargs):
        pass


class _FakeModel:
    """Stands in for the PPO model; `logger` is read off it by BaseCallback."""

    def __init__(self):
        self.saved_to = []
        self.logger = _FakeLogger()
        self.num_timesteps = 5

    def save(self, path):
        self.saved_to.append(path)


def _callback_at_eval(monkeypatch, phillips, selfplay, best_key=None):
    """Run one evaluation tick and report what got saved where."""
    callback = train.TqdmCallback(
        total_timesteps=10,
        opponents=[train.phillips_opponent()],
        primary_name="Phillips",
        best_model_path="/tmp/best",
        opponent_model_path="/tmp/opponent",
        model_prefix="test",
        eval_every=1,
        save_every=10**9,
    )
    if best_key is not None:
        callback._best_key = best_key

    results = {"Phillips": phillips}
    monkeypatch.setattr(
        train, "evaluate_opponent", lambda model, opp, **kw: results[opp.name]
    )
    monkeypatch.setattr(
        train,
        "load_selfplay_opponent",
        lambda path: (
            None
            if selfplay is None
            else train.Opponent(name="selfplay", make_opponent=lambda pid: None)
        ),
    )
    if selfplay is not None:
        results["selfplay"] = selfplay

    model = _FakeModel()
    callback.model = model
    callback.num_timesteps = model.num_timesteps
    callback.pbar = type(
        "P",
        (),
        {
            "n": 0,
            "update": lambda self, *a: None,
            "write": lambda self, *a: None,
            "set_description": lambda self, *a: None,
        },
    )()
    callback._on_step()
    return callback, model.saved_to


def _result(winrate, margin=0.0):
    return {
        "winrate": winrate,
        "margin": margin,
        "margin_ci": 0.0,
        "clears": 0.0,
        "rl_avg": 0.0,
        "opp_avg": 0.0,
    }


def test_partner_is_promoted_when_we_beat_it(monkeypatch):
    _, saved = _callback_at_eval(
        monkeypatch, phillips=_result(10), selfplay=_result(90)
    )

    assert "/tmp/opponent" in saved


def test_partner_is_not_promoted_when_we_lose_to_it(monkeypatch):
    _, saved = _callback_at_eval(
        monkeypatch, phillips=_result(80), selfplay=_result(40)
    )

    assert "/tmp/opponent" not in saved


def test_partner_is_promoted_despite_a_terrible_primary_result(monkeypatch):
    """Partner beaten, primary win rate far below its own high-water mark."""
    _, saved = _callback_at_eval(
        monkeypatch,
        phillips=_result(5, margin=-80.0),
        selfplay=_result(100),
        best_key=(48.0, -10.0),  # a much better Phillips result already banked
    )

    assert "/tmp/opponent" in saved, "a bad Phillips score must not block promotion"
    assert "/tmp/best" not in saved, "but it must not count as a new best either"


def test_first_eval_seeds_a_partner_when_none_exists(monkeypatch):
    _, saved = _callback_at_eval(monkeypatch, phillips=_result(3), selfplay=None)

    assert "/tmp/opponent" in saved


def test_best_is_saved_on_primary_improvement_only(monkeypatch):
    _, saved = _callback_at_eval(
        monkeypatch,
        phillips=_result(60),
        selfplay=_result(20),  # losing to the partner, so no promotion
        best_key=(50.0, 0.0),
    )

    assert "/tmp/best" in saved
    assert "/tmp/opponent" not in saved


def test_promotion_threshold_is_the_boundary(monkeypatch):
    _, at = _callback_at_eval(
        monkeypatch, phillips=_result(10), selfplay=_result(train.PROMOTION_WINRATE)
    )
    _, below = _callback_at_eval(
        monkeypatch,
        phillips=_result(10),
        selfplay=_result(train.PROMOTION_WINRATE - 1),
    )

    assert "/tmp/opponent" in at
    assert "/tmp/opponent" not in below
