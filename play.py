"""
GoAcademics - play.py
----------------------
Entry point for running the best-performing trained RL agent in the
GoAcademics classroom simulation with pygame GUI and terminal verbose output.

This is the primary demonstration script referenced in the project rubric.

Usage:
    python play.py                          # auto-detect best model
    python play.py --algo dqn               # run best DQN model
    python play.py --algo ppo               # run best PPO model
    python play.py --algo a2c               # run best A2C model
    python play.py --algo reinforce         # run best REINFORCE model
    python play.py --episodes 3             # run N episodes
    python play.py --students 15            # classroom size
    python play.py --no-render              # terminal only (no GUI)
"""

import sys
import os

# Re-use all logic from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    main()
