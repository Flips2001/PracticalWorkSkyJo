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
