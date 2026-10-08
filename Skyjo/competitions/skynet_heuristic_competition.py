"""Our RL agent and Phillips against SkyNet's six heuristic players.

SkyNet (arXiv:2603.27751) was trained and evaluated against these rule-based
players; see `skynet_bridge` for how the games are run. The SkyNet code is
downloaded into `.skynet/` on the first run.
"""

from Skyjo.competitions import skynet_bridge


def main():
    skynet_bridge.download()
    from heuristic_bots import BOT_REGISTRY

    with skynet_bridge.worker_pool() as pool:
        for player in ("rl", "phillips"):
            rows = [
                skynet_bridge.play_matchup(pool, player, bot) for bot in BOT_REGISTRY
            ]
            skynet_bridge.print_results(player, rows)


if __name__ == "__main__":
    main()
