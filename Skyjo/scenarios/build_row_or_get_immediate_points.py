from Skyjo.scenarios.scenario import Scenario, run_scenario

SCENARIO = Scenario(
    name="Build row or get immediate points?",
    description="""
    Does the agent take the 4 and start building a column, or does it replace the 9
    """,
    your_grid=[
        [2, 2, "?", 9],
        ["?", "?", 4, "?"],
        ["?", "?", "?", "?"],
    ],
    opponent_grid=[
        ["?", "?", "?", 2],
        ["?", 2, 4, "?"],
        ["?", "?", "?", 7],
    ],
    discard=[4],
    first_player="agent",
    seed=12,
)

if __name__ == "__main__":
    run_scenario(SCENARIO)
