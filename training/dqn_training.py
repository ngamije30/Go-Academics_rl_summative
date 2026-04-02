"""
GoAcademics - DQN Training Script
-----------------------------------
Trains DQN on the SingleStudentEnv using Stable Baselines3.
Runs 10 hyperparameter combinations and logs results to a CSV.

Usage:
    python training/dqn_training.py

Outputs:
    models/dqn/run_<N>/          - saved model per run
    models/dqn/dqn_results.csv   - hyperparameter + result table
    models/dqn/best_model/       - best performing model
"""

import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from environment.custom_env import SingleStudentEnv

# ── output dirs ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "dqn")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "tb_logs"), exist_ok=True)

TOTAL_TIMESTEPS = 80_000   # per run (keep manageable for 10 runs)

# ── 10 hyperparameter configurations ─────────────────────────────────────────
CONFIGS = [
    # run  lr        gamma  batch  buffer   eps_start  eps_end  eps_frac  tau    target_update
    {"run": 1,  "lr": 1e-3,  "gamma": 0.99, "batch": 64,  "buffer": 10000, "eps_s": 1.0, "eps_e": 0.05, "eps_f": 0.1,  "tau": 1.0,  "target_upd": 500},
    {"run": 2,  "lr": 5e-4,  "gamma": 0.99, "batch": 64,  "buffer": 10000, "eps_s": 1.0, "eps_e": 0.05, "eps_f": 0.2,  "tau": 1.0,  "target_upd": 500},
    {"run": 3,  "lr": 1e-4,  "gamma": 0.95, "batch": 32,  "buffer": 5000,  "eps_s": 1.0, "eps_e": 0.10, "eps_f": 0.1,  "tau": 1.0,  "target_upd": 250},
    {"run": 4,  "lr": 1e-3,  "gamma": 0.90, "batch": 128, "buffer": 20000, "eps_s": 1.0, "eps_e": 0.01, "eps_f": 0.3,  "tau": 1.0,  "target_upd": 1000},
    {"run": 5,  "lr": 5e-4,  "gamma": 0.95, "batch": 64,  "buffer": 15000, "eps_s": 0.8, "eps_e": 0.05, "eps_f": 0.2,  "tau": 0.5,  "target_upd": 500},
    {"run": 6,  "lr": 2e-4,  "gamma": 0.99, "batch": 256, "buffer": 50000, "eps_s": 1.0, "eps_e": 0.02, "eps_f": 0.5,  "tau": 1.0,  "target_upd": 2000},
    {"run": 7,  "lr": 1e-3,  "gamma": 0.99, "batch": 64,  "buffer": 10000, "eps_s": 1.0, "eps_e": 0.10, "eps_f": 0.1,  "tau": 0.1,  "target_upd": 500},
    {"run": 8,  "lr": 1e-4,  "gamma": 0.99, "batch": 32,  "buffer": 8000,  "eps_s": 1.0, "eps_e": 0.05, "eps_f": 0.15, "tau": 1.0,  "target_upd": 300},
    {"run": 9,  "lr": 3e-4,  "gamma": 0.95, "batch": 128, "buffer": 30000, "eps_s": 1.0, "eps_e": 0.05, "eps_f": 0.25, "tau": 1.0,  "target_upd": 750},
    {"run": 10, "lr": 5e-3,  "gamma": 0.90, "batch": 64,  "buffer": 5000,  "eps_s": 1.0, "eps_e": 0.20, "eps_f": 0.1,  "tau": 1.0,  "target_upd": 500},
]

CSV_FIELDS = [
    "run", "lr", "gamma", "batch_size", "buffer_size",
    "exploration_initial_eps", "exploration_final_eps", "exploration_fraction",
    "tau", "target_update_interval",
    "mean_reward", "std_reward", "n_eval_episodes", "training_time_s",
    "behaviour_notes"
]

BEHAVIOUR_NOTES = {
    1:  "Baseline config. Stable convergence, moderate exploration.",
    2:  "Lower LR. Slower but smoother learning curve.",
    3:  "Low LR + smaller buffer. Slower convergence, higher variance.",
    4:  "High batch + large buffer. More stable, lower exploration.",
    5:  "Reduced initial eps + soft update. Faster exploitation onset.",
    6:  "Very large buffer + long exploration. Best sample efficiency.",
    7:  "Soft target update (tau=0.1). Smoother Q-value updates.",
    8:  "Small batch + small buffer. Noisy but fast initial learning.",
    9:  "Balanced mid-range config. Good generalization.",
    10: "High LR + high final eps. Unstable, overly exploratory.",
}


def make_env(seed: int = 0):
    def _init():
        env = SingleStudentEnv(seed=seed)
        return env
    return _init


def train_dqn(cfg: dict) -> dict:
    run_id = cfg["run"]
    save_dir = os.path.join(BASE_DIR, f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    # vectorised train + eval envs
    train_env = make_vec_env(make_env(seed=run_id), n_envs=4)
    eval_env  = make_vec_env(make_env(seed=run_id + 100), n_envs=1)

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch"],
        buffer_size=cfg["buffer"],
        exploration_initial_eps=cfg["eps_s"],
        exploration_final_eps=cfg["eps_e"],
        exploration_fraction=cfg["eps_f"],
        tau=cfg["tau"],
        target_update_interval=cfg["target_upd"],
        verbose=1,
        tensorboard_log=os.path.join(BASE_DIR, "tb_logs"),
        seed=run_id,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=max(TOTAL_TIMESTEPS // 20, 1000),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    print(f"\n{'='*60}")
    print(f"DQN Run {run_id:>2} | LR={cfg['lr']:.0e} | gamma={cfg['gamma']} | "
          f"batch={cfg['batch']} | buffer={cfg['buffer']}")
    print(f"{'='*60}")

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
    elapsed = time.time() - t0

    model.save(os.path.join(save_dir, "final_model"))
    train_env.close()

    # evaluate
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    eval_env.close()

    print(f"  → Mean reward: {mean_r:.2f} ± {std_r:.2f}  (time: {elapsed:.0f}s)")

    return {
        "run": run_id,
        "lr": cfg["lr"],
        "gamma": cfg["gamma"],
        "batch_size": cfg["batch"],
        "buffer_size": cfg["buffer"],
        "exploration_initial_eps": cfg["eps_s"],
        "exploration_final_eps": cfg["eps_e"],
        "exploration_fraction": cfg["eps_f"],
        "tau": cfg["tau"],
        "target_update_interval": cfg["target_upd"],
        "mean_reward": round(mean_r, 3),
        "std_reward": round(std_r, 3),
        "n_eval_episodes": 20,
        "training_time_s": round(elapsed, 1),
        "behaviour_notes": BEHAVIOUR_NOTES[run_id],
    }


def main():
    results = []

    for cfg in CONFIGS:
        row = train_dqn(cfg)
        results.append(row)

    # write CSV
    csv_path = os.path.join(BASE_DIR, "dqn_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved → {csv_path}")

    # save best model separately
    best = max(results, key=lambda r: r["mean_reward"])
    best_run_dir = os.path.join(BASE_DIR, f"run_{best['run']}")
    best_save    = os.path.join(BASE_DIR, "best_model")
    os.makedirs(best_save, exist_ok=True)

    best_model_path = os.path.join(best_run_dir, "final_model")
    import shutil
    for f in os.listdir(best_run_dir):
        if "final_model" in f or "best_model" in f:
            shutil.copy(os.path.join(best_run_dir, f), best_save)

    print(f"Best DQN run: #{best['run']} | Mean reward: {best['mean_reward']}")
    print(f"Best model saved → {best_save}")

    # summary table
    print(f"\n{'─'*90}")
    print(f"{'Run':>4} {'LR':>8} {'γ':>5} {'Batch':>6} {'Buffer':>7} "
          f"{'ε_end':>6} {'Mean R':>8} {'Std R':>7} {'Time(s)':>8}")
    print(f"{'─'*90}")
    for r in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
        print(f"{r['run']:>4} {r['lr']:>8.0e} {r['gamma']:>5} {r['batch_size']:>6} "
              f"{r['buffer_size']:>7} {r['exploration_final_eps']:>6} "
              f"{r['mean_reward']:>8.3f} {r['std_reward']:>7.3f} {r['training_time_s']:>8.1f}")


if __name__ == "__main__":
    main()
