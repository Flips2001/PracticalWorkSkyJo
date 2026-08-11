"""The agent must choose between a column clear and replacing a 12."""

from Skyjo.scenarios.scenario import Scenario, run_scenario

SCENARIO = Scenario(
    name="Clear or dump",
    description="""
    The RL agent can take the 3 from the discard pile and place it in either of
    two attractive slots. One placement completes its column of 3s; the other
    replaces its visible 12.
    """,
    your_grid=[
        [4, "?", 6, 2],
        ["?", 0, "?", 7],
        [1, "?", 5, "?"],
    ],
    opponent_grid=[
        [8, 3, "?", 1],
        [5, 3, 4, "?"],
        [12, "?", "?", 2],
    ],
    discard=[11, 9, 3],
    first_player="agent",
    seed=8,
)

if __name__ == "__main__":
    run_scenario(SCENARIO)
