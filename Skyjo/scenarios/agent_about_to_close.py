"""The agent can end the round now or spend a turn improving its score."""

from Skyjo.scenarios.scenario import Scenario, run_scenario

SCENARIO = Scenario(
    name="Finish or improve?",
    description="""
    The RL agent shows 28 points and has one face-down card left. Its opponent
    shows 51 with four cards still hidden. A 9 is available on the discard pile:
    the agent can put it into its last hidden slot and trigger the final turn,
    or replace a visible card such as its 12 and keep the round going.
    """,
    your_grid=[
        [12, "?", 8, "?"],
        [6, 9, "?", 4],
        ["?", 3, 7, 2],
    ],
    opponent_grid=[
        [2, 1, 0, 3],
        [1, 4, 2, 0],
        ["?", 2, 1, 12],
    ],
    discard=[8, 11, 9],
    first_player="agent",
    seed=5,
)

if __name__ == "__main__":
    run_scenario(SCENARIO)
