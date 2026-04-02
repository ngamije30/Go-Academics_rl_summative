# GoAcademics — RL Summative Assignment

## Problem Statement

Most Rwandan secondary schools, especially in rural areas, lack reliable internet
and Learning Management Systems, making existing predictive analytics tools
unusable in these contexts. Teachers have no structured way to identify at-risk
students early enough to intervene before the end-of-term examinations.

**GoAcademics** addresses this by providing an offline, teacher-centred early
warning system that uses simple, readily available data — attendance, continuous
assessment (CA) scores, and study hours — to identify at-risk students early and
recommend targeted interventions. A Reinforcement Learning agent is trained to
act as the intervention-recommendation engine: each week it observes each
student's academic indicators and decides which support action to apply
(e.g. peer tutoring, parent alert, counselling) to maximise the end-of-term
pass rate while preventing dropout.

---

## Project Structure

```
ngamije_rl_summative/
├── environment/
│   ├── custom_env.py      # GoAcademicsEnv (multi-student) + SingleStudentEnv
│   ├── rendering.py       # pygame classroom dashboard + random-agent demo
│   └── __init__.py
├── training/
│   ├── dqn_training.py    # DQN — 10 hyperparameter runs
│   └── pg_training.py     # REINFORCE, PPO, A2C — 10 runs each
├── models/
│   ├── dqn/               # Saved DQN models + dqn_results.csv
│   ├── pg/                # Saved PG models (reinforce/, ppo/, a2c/) + CSVs
│   ├── learning_curves.png
│   ├── all_algorithms_comparison.png
│   ├── best_run_comparison.png
│   ├── best_agent_episode.png
│   ├── lr_sensitivity.png
│   └── random_agent_baseline.png
├── main.py                # Entry point for running best performing model
├── play.py                # Alias for main.py (rubric entry point)
├── random_agent.py        # Random agent demo — no model required
├── requirements.txt
└── README.md
```

---

## Quick Start

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. Tested with torch 2.5.1 (CPU), stable-baselines3 2.8.0, pygame 2.6.1.

**Step 2 — Run the GUI simulation with the best trained agent:**
```bash
python play.py
```

That's it. The pre-trained models are included in the `models/` folder — no training required to run the demo.

---

## Run the random-agent demo (no training needed)

```bash
python random_agent.py
```

Opens the pygame classroom dashboard with a fully random agent — demonstrates
the environment visualization without any trained model.

Options:
```bash
python random_agent.py --students 15    # custom classroom size
python random_agent.py --episodes 3     # more episodes
python random_agent.py --no-render      # terminal only
python random_agent.py --seed 42        # fixed seed
```

---

## Train all models

All models were originally trained and analysed in the Jupyter notebook
`model_training.ipynb`. To reproduce training from scratch you can either:

**Option A — Run the notebook (recommended):**
Open `model_training.ipynb` in Jupyter and run all cells. The notebook covers
environment setup, all 4 algorithms (10 runs each), evaluation, and plot generation.

**Option B — Run the training scripts directly:**
```bash
# DQN — 10 hyperparameter runs (~80k timesteps each)
python training/dqn_training.py

# REINFORCE, PPO, A2C — 10 runs each (~150k timesteps each)
python training/pg_training.py
```

Results CSVs are written to `models/dqn/` and `models/pg/`.
Best models are saved automatically to `models/dqn/best_model/` and `models/pg/best_model/`.

---

## Run the best agent

```bash
python play.py                        # auto-detect best model
python play.py --algo ppo             # force PPO
python play.py --algo dqn             # force DQN
python play.py --algo a2c             # force A2C
python play.py --algo reinforce       # force REINFORCE
python play.py --episodes 5           # run 5 episodes
python play.py --students 20          # 20-student classroom
python play.py --no-render            # terminal only (no GUI)
```

The pygame window stays open after all episodes complete — press any key or close the window to exit.

---

## Environment Details

| Component | Description |
|---|---|
| **Mission** | Offline early warning system for at-risk secondary school students |
| **Observation** | 6 features per student: attendance rate, CA score (normalised), study hours, assignment completion, term week, risk level |
| **Actions** | 6 discrete: No action, Parent alert, Peer tutoring, Counselling, Extra CA, Escalate |
| **Rewards** | +10 risk-drop, +20 pass, -20 dropout, -5 risk-rise, -1 ignoring high-risk, -2 unnecessary escalation |
| **Terminal** | Week 13 reached OR all students resolved (passed or dropped out) |
| **Visualisation** | pygame classroom dashboard — colour-coded student cards (green=low risk, orange=medium, red=high/dropped, blue=passed) |

---

## Algorithm Notes

| Algorithm | Environment | Best Mean Reward |
|---|---|---|
| DQN | SingleStudentEnv (Discrete action space) | ~35 (1 student) |
| REINFORCE | GoAcademicsEnv 10-student (via A2C, vf_coef=0) | ~1867 |
| PPO | GoAcademicsEnv 10-student | ~1886 |
| A2C | GoAcademicsEnv 10-student | ~1916 |

> **Note:** DQN rewards are not directly comparable to PG method rewards because DQN
> trains on a single-student wrapper (required for Discrete action spaces), while PPO/A2C/REINFORCE
> operate on the full 10-student classroom. A2C achieved the highest reward and is selected as
> the default best model.

---

## Results

Training results for all 10 runs per algorithm are saved as CSV files:

- `models/dqn/dqn_results.csv`
- `models/pg/reinforce/reinforce_results.csv`
- `models/pg/ppo/ppo_results.csv`
- `models/pg/a2c/a2c_results.csv`

Key plots saved in `models/`:

- **learning_curves.png** — per-run reward curves for all 4 algorithms
- **all_algorithms_comparison.png** — mean reward per hyperparameter run
- **best_run_comparison.png** — best run per algorithm comparison
- **best_agent_episode.png** — best A2C agent episode trace (CA, attendance, risk, cumulative reward)
- **lr_sensitivity.png** — learning rate sensitivity analysis per algorithm
- **random_agent_baseline.png** — random agent baseline for comparison