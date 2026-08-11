"""The agent must choose between a column clear and replacing a 12."""

from Skyjo.scenarios.scenario import Scenario, run_scenario

SCENARIO = Scenario(
    name="Build and remove column",
    description="""
    The RL agent can build and then remove a colum
    """,
    your_grid=[
        [4, "?", 6, 2],
        ["?", 0, "?", 7],
        [1, "?", 5, "?"],
    ],
    opponent_grid=[
        [8, 3, "?", 1],
        [5, "?", 4, "?"],
        [12, "?", "?", 2],
    ],
    discard=[11, 9, 3],
    draw=[3, 3, 3],
    first_player="agent",
    seed=8,
)

if __name__ == "__main__":
    run_scenario(SCENARIO)
