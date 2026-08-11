"""A deliberately counter-intuitive high-value column clear for the agent."""

from Skyjo.scenarios.scenario import Scenario, run_scenario

SCENARIO = Scenario(
    name="Clear the column",
    description="""
    The RL agent has two visible 6s in one column and a hidden third slot. A 6
    is on top of the discard pile, so taking the worst individual card in Skyjo
    can complete the set and remove the whole column.
    """,
    your_grid=[
        [5, "?", 3, "?"],
        ["?", 2, "?", 8],
        [7, "?", "?", 1],
    ],
    opponent_grid=[
        [4, "?", 1, 6],
        ["?", 6, "?", 6],
        [0, "?", "?", "?"],
    ],
    discard=[7, 10, 1, 6],
    first_player="agent",
    seed=1,
)

if __name__ == "__main__":
    run_scenario(SCENARIO)
