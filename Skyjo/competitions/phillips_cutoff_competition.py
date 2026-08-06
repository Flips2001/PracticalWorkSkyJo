from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from multiprocessing import get_context
import os
import random
from typing import Optional

from tqdm import tqdm

from Skyjo.src.players.phillips_player import PhillipsPlayer
from Skyjo.src.skyjo_game import SkyjoGame

CUTOFFS = list(range(-2, 13))
MAX_AUTO_WORKERS = 16

PairingJob = tuple[int, int, int, Optional[int]]
PairingResult = tuple[int, int, int, int, int]


def play_pairing(cutoff_a, cutoff_b, num_games, seed=None):
    """Play a series of games between two cutoff configs, alternating seats.

    Returns:
        Tuple ``(wins_a, wins_b, ties)``.
    """
    if seed is not None:
        random.seed(seed)

    wins_a = wins_b = ties = 0
    for i in range(num_games):
        seat_a = i % 2
        seat_b = 1 - seat_a

        seats: list = [None, None]
        seats[seat_a] = PhillipsPlayer(seat_a, f"cutoff={cutoff_a}", cutoff=cutoff_a)
        seats[seat_b] = PhillipsPlayer(seat_b, f"cutoff={cutoff_b}", cutoff=cutoff_b)

        game = SkyjoGame()
        for player in seats:
            game.add_player(player)
        game.play_game()

        scores = game.game_state.all_player_final_scores
        score_a, score_b = scores[seat_a], scores[seat_b]
        if score_a < score_b:
            wins_a += 1
        elif score_b < score_a:
            wins_b += 1
        else:
            ties += 1
    return wins_a, wins_b, ties


def _play_pairing_job(job: PairingJob) -> PairingResult:
    """Run one independently seeded pairing inside a worker process."""
    cutoff_a, cutoff_b, num_games, seed = job
    wins_a, wins_b, ties = play_pairing(cutoff_a, cutoff_b, num_games, seed=seed)
    return cutoff_a, cutoff_b, wins_a, wins_b, ties


def resolve_worker_count(requested_workers: Optional[int], job_count: int) -> int:
    """Choose a useful worker count while leaving one CPU available by default."""
    if requested_workers is not None and requested_workers < 1:
        raise ValueError("workers must be at least 1")
    if job_count < 1:
        return 1
    if requested_workers is not None:
        return min(requested_workers, job_count)

    cpu_count = os.cpu_count() or 1
    available_cpus = max(1, cpu_count - 1)
    return min(job_count, available_cpus, MAX_AUTO_WORKERS)


def play_pairings(
    cutoffs,
    num_games,
    workers=None,
    seed=None,
    show_progress=True,
) -> list[PairingResult]:
    """Play every cutoff pairing, optionally distributing them across processes."""
    pairs = list(combinations(cutoffs, 2))
    worker_count = resolve_worker_count(workers, len(pairs))
    jobs = [
        (a, b, num_games, None if seed is None else seed + index)
        for index, (a, b) in enumerate(pairs)
    ]
    progress = tqdm(
        total=len(jobs),
        desc=f"Pairings ({worker_count} workers)",
        unit="pairing",
        disable=not show_progress,
    )

    try:
        if worker_count == 1:
            results = []
            for job in jobs:
                results.append(_play_pairing_job(job))
                progress.update()
            return results

        results = []
        context = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = {executor.submit(_play_pairing_job, job): job[:2] for job in jobs}
            for future in as_completed(futures):
                cutoff_a, cutoff_b = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"Pairing {cutoff_a} vs {cutoff_b} failed"
                    ) from exc
                progress.update()
        return results
    finally:
        progress.close()


def run_cutoff_competition(num_games, workers=None, seed=None):
    """Run the complete cutoff tournament and print its results."""
    if num_games < 1:
        raise ValueError("num_games must be at least 1")

    cutoffs = CUTOFFS
    # wins[a][b] = games cutoff a won against cutoff b
    wins = {a: {b: 0 for b in cutoffs} for a in cutoffs}
    total_wins = {a: 0 for a in cutoffs}
    total_games = {a: 0 for a in cutoffs}

    pair_count = len(list(combinations(cutoffs, 2)))
    worker_count = resolve_worker_count(workers, pair_count)
    print(
        f"Running {pair_count * num_games:,} games across "
        f"{worker_count} worker{'s' if worker_count != 1 else ''}."
    )

    results = play_pairings(
        cutoffs,
        num_games,
        workers=worker_count,
        seed=seed,
    )
    for a, b, wins_a, wins_b, _ in results:
        wins[a][b] = wins_a
        wins[b][a] = wins_b
        total_wins[a] += wins_a
        total_wins[b] += wins_b
        total_games[a] += num_games
        total_games[b] += num_games

    print(f"\nWins per pairing ({num_games} games each, row vs column):")
    header = "cutoff |" + "".join(f"{b:>6}" for b in cutoffs)
    print(header)
    print("-" * len(header))
    for a in cutoffs:
        cells = "".join("     -" if a == b else f"{wins[a][b]:>6}" for b in cutoffs)
        print(f"{a:>6} |{cells}")

    print("\nOverall win rates:")
    for a in sorted(cutoffs, key=lambda c: total_wins[c], reverse=True):
        rate = total_wins[a] / total_games[a] * 100
        print(f"  cutoff={a}: {total_wins[a]}/{total_games[a]} wins ({rate:.1f}%)")

    best = max(cutoffs, key=lambda c: total_wins[c])
    print(f"\nBest config: cutoff={best}")


if __name__ == "__main__":
    games_per_pairing = 1000
    run_cutoff_competition(games_per_pairing, 16)
