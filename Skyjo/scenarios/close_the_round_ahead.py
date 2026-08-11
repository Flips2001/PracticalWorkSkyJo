"""A controlled test of whether the agent closes while far ahead."""

from Skyjo.scenarios.scenario import Scenario, run_scenario

SCENARIO = Scenario(
    name="Closing with a lead",
    description="""
    The RL agent has one hidden card left and shows only 11 points. Its opponent
    already shows 42 and still has five hidden cards. The agent can end the round
    this turn, although the open 7 and the next hidden draw both make the exact
    route to finishing non-trivial.
    """,
    your_grid=[
        [8, "?", 6, 4],
        ["?", 9, "?", 5],
        [7, 3, "?", "?"],
    ],
    opponent_grid=[
        [1, 0, 2, -1],
        [3, 1, 0, 2],
        [2, "6?", 1, 0],
    ],
    discard=[10, 12, 7],
    draw=[9],
    first_player="agent",
    seed=3,
)

if __name__ == "__main__":
    run_scenario(SCENARIO)
