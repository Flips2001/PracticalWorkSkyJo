# PracticalWorkSkyJo

## Installation
1. use python 3.12 to create a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
    ```
   
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Install precommit hooks:
    ```bash
   pre-commit install
   ```

## Play a game
Run the `main.py` file to play a game of Skyjo. To change the number or type of players, add, change, or remove the player declarations and the `add_player()` calls.

Press `a` in-game to toggle analyze mode and inspect the RL player one action
at a time. While enabled, each RL sub-action pauses until Enter is pressed,
and the terminal UI shows the decision-time board with an integrated-gradients
heatmap for that move (green = little influence, red = much influence).

## Alternate user interfaces

`SkyjoGame` also exposes UI-neutral lifecycle hooks. `before_action` and
`after_action` surround every legal sub-action, `round_started` exposes the
newly dealt round and opening discard, and `round_scored` exposes the scored
board before it is reset. A UI can block in these hooks while it animates, then
resume the same authoritative `play_game()` loop. Players still select from the
legal actions supplied by Skyjo, so an alternate UI does not need its own copy
of the rules.

## Observe the agent in a scenario
A scenario is a hand-built position - your grid, the agent's grid, the discard
pile and optionally the next cards off the draw pile - for probing a particular
RL behavior in the same terminal UI. Every included scenario starts with the
agent's decision. Its briefing explains why that decision is interesting to
observe. Analyze mode starts enabled so each agent action pauses with its
integrated-gradients explanation; after the opening move, you play the opposing
side normally. Press `a` if you want to turn analysis off.

Each scenario lives in its own file under `Skyjo/scenarios/` and is runnable on
its own:

```bash
python -m Skyjo.scenarios.column_clear_with_twelves
```

List what is there:

```bash
python -m Skyjo.scenarios
```

By default a scenario plays the one round from its position and then reports the
score; add `--full-game` to carry on with normally dealt rounds until someone
reaches 100. `--seed` re-rolls the face-down cards the scenario left open and
`--model` picks a different checkpoint.

### Writing one
Copy any file in `Skyjo/scenarios/` and change the data. Put the decision under
test on `opponent_grid` (the RL agent's board), and leave `first_player` as
`"agent"`. Describe the competing incentives and why the policy's choice is
informative rather than instructing the human which move to make.

Grid entries are `5` for a face-up 5, `"5?"` for a face-down card that turns out
to be a 5, and `"?"` for a face-down card whose value is drawn from what is left
in the deck. Rows shorter than four cards model a column that was already
cleared. Every named card is taken out of one real 150 card deck, so an impossible
position (six -2s, a column the game would clear immediately) is rejected with an
explanation instead of quietly distorting the counts the agent reasons about.
