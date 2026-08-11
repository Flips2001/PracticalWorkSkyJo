"""A negative-value column tests whether the agent avoids an attractive trap."""

from Skyjo.scenarios.scenario import Scenario, run_scenario

SCENARIO = Scenario(
    name="The tempting minus two",
    description="""
    The RL agent has two visible -2s in one column and a hidden third slot. A -2
    on the discard pile could complete and clear that column, but removing three
    -2s would throw away six points of beneficial negative score.
    """,
    your_grid=[
        [6, "?", 2, "?"],
        ["?", 4, "?", 3],
        [8, "?", 1, "?"],
    ],
    opponent_grid=[
        [-2, 5, "?", 9],
        [-2, "?", 4, "?"],
        ["?", 3, "?", 6],
    ],
    discard=[12, 9, -2],
    first_player="agent",
    seed=2,
)

if __name__ == "__main__":
    run_scenario(SCENARIO)
