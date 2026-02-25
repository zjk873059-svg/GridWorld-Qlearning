# GridWorld-Qlearning
CDS524 Assignment 1 – Reinforcement Learning Game
# Grid World Q-learning

This is a reinforcement learning project for CDS524 Assignment 1.  
It implements a Q-learning agent that learns to collect treasures and avoid traps in a 6×6 grid world.  
The game is built with Pygame and the Q-learning algorithm uses a table-based approach.

## Game Rules

- **Grid size**: 6×6
- **Start**: Green cell (top-left)
- **Treasures**: Yellow cells (+30 reward each, +50 bonus when all collected)
- **Traps**: Red cells (-30 reward, episode ends)
- **Empty cells**: -0.2 step penalty to encourage efficiency
- **Actions**: Up, Down, Left, Right
- **Goal**: Collect all treasures while avoiding traps.

## Dependencies

- Python 3.9+
- pygame
- numpy

Install dependencies with:

```bash
pip install -r requirements.txt
