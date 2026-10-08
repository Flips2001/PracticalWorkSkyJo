"""Our RL agent against SkyNet (arXiv:2603.27751), the belief-aware MuZero itself.

Uses the released iteration-1000 checkpoint; see `skynet_bridge` for how the
games are run. The SkyNet code and its 216 MB checkpoint are downloaded into
`.skynet/` on the first run. SkyNet searches 200 simulations per decision, so
the 1000 games take roughly 1.5 hours on 16 cores.
"""

from Skyjo.competitions import skynet_bridge


def main():
    skynet_bridge.download(checkpoint=True)
    with skynet_bridge.worker_pool(with_skynet=True) as pool:
        skynet_bridge.print_results(
            "rl", [skynet_bridge.play_matchup(pool, "rl", "skynet")]
        )


if __name__ == "__main__":
    main()
