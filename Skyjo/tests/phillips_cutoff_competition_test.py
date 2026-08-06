import pytest

from Skyjo.competitions import phillips_cutoff_competition as competition


def test_resolve_worker_count_uses_available_cpus(monkeypatch):
    monkeypatch.setattr(competition.os, "cpu_count", lambda: 16)

    assert competition.resolve_worker_count(None, job_count=105) == 15
    assert competition.resolve_worker_count(8, job_count=105) == 8
    assert competition.resolve_worker_count(20, job_count=3) == 3


def test_resolve_worker_count_rejects_non_positive_values():
    with pytest.raises(ValueError, match="at least 1"):
        competition.resolve_worker_count(0, job_count=10)


def test_play_pairing_is_reproducible_with_a_seed():
    first = competition.play_pairing(-1, 2, num_games=3, seed=12345)
    second = competition.play_pairing(-1, 2, num_games=3, seed=12345)

    assert first == second


def test_play_pairings_assigns_stable_seed_per_pairing(monkeypatch):
    seen_jobs = []

    def fake_play_pairing_job(job):
        seen_jobs.append(job)
        cutoff_a, cutoff_b, _, _ = job
        return cutoff_a, cutoff_b, 1, 2, 3

    monkeypatch.setattr(competition, "_play_pairing_job", fake_play_pairing_job)

    results = competition.play_pairings(
        cutoffs=[-2, -1, 0],
        num_games=7,
        workers=1,
        seed=100,
        show_progress=False,
    )

    assert seen_jobs == [
        (-2, -1, 7, 100),
        (-2, 0, 7, 101),
        (-1, 0, 7, 102),
    ]
    assert results == [
        (-2, -1, 1, 2, 3),
        (-2, 0, 1, 2, 3),
        (-1, 0, 1, 2, 3),
    ]
