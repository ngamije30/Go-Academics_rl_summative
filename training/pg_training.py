"""
GoAcademics - Policy Gradient Training Script
----------------------------------------------
Trains three policy gradient algorithms on GoAcademicsEnv (multi-student):
  - REINFORCE  (via SB3 A2C with n_steps=episode_len, no value baseline)
  - PPO        (Proximal Policy Optimization)
  - A2C        (Advantage Actor-Critic)

Each algorithm gets 10 hyperparameter runs.
Results are saved per-algorithm CSV and a combined summary.

Usage:
    python training/pg_training.py

Outputs:
    models/pg/reinforce/   - saved REINFORCE models + CSV
    models/pg/ppo/         - saved PPO models + CSV
    models/pg/a2c/         - saved A2C models + CSV
    models/pg/best_model/  - best model across all PG algorithms
"""

import os
import sys
import csv
import time
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import GoAcademicsEnv

# ── output dirs ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pg")
for sub in ["reinforce", "ppo", "a2c", "best_model", "tb_logs"]:
    os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

TOTAL_TIMESTEPS = 150_000   # per run
N_STUDENTS = 10


# ── env factory ───────────────────────────────────────────────────────────────
def make_classroom_env(seed: int = 0, n_students: int = N_STUDENTS):
    def _init():
        return GoAcademicsEnv(n_students=n_students, render_mode=None, seed=seed)
    return _init


# ═════════════════════════════════════════════════════════════════════════════
# REINFORCE  (approximated via A2C with vf_coef=0 and large n_steps)
# SB3 does not ship a standalone REINFORCE, so we use A2C configured as a
# pure policy-gradient estimator: no value baseline (vf_coef=0), full
# episode rollouts (n_steps = TERM_LENGTH), no advantage normalisation.
# ═════════════════════════════════════════════════════════════════════════════
REINFORCE_CONFIGS = [
    {"run":1,  "lr":1e-3, "gamma":0.99, "n_steps":13, "ent_coef":0.01, "vf_coef":0.0, "max_grad":0.5},
    {"run":2,  "lr":5e-4, "gamma":0.99, "n_steps":13, "ent_coef":0.02, "vf_coef":0.0, "max_grad":0.5},
    {"run":3,  "lr":1e-4, "gamma":0.95, "n_steps":13, "ent_coef":0.01, "vf_coef":0.0, "max_grad":0.5},
    {"run":4,  "lr":3e-3, "gamma":0.99, "n_steps":13, "ent_coef":0.00, "vf_coef":0.0, "max_grad":0.5},
    {"run":5,  "lr":1e-3, "gamma":0.90, "n_steps":13, "ent_coef":0.05, "vf_coef":0.0, "max_grad":0.5},
    {"run":6,  "lr":2e-4, "gamma":0.99, "n_steps":26, "ent_coef":0.01, "vf_coef":0.0, "max_grad":1.0},
    {"run":7,  "lr":1e-3, "gamma":0.99, "n_steps":13, "ent_coef":0.01, "vf_coef":0.0, "max_grad":0.1},
    {"run":8,  "lr":5e-4, "gamma":0.95, "n_steps":13, "ent_coef":0.10, "vf_coef":0.0, "max_grad":0.5},
    {"run":9,  "lr":1e-3, "gamma":0.99, "n_steps":52, "ent_coef":0.01, "vf_coef":0.0, "max_grad":0.5},
    {"run":10, "lr":5e-3, "gamma":0.90, "n_steps":13, "ent_coef":0.00, "vf_coef":0.0, "max_grad":2.0},
]

REINFORCE_NOTES = {
    1:  "Baseline REINFORCE. Stable but high variance due to no baseline.",
    2:  "Lower LR. Slower convergence, reduced variance.",
    3:  "Low LR + lower gamma. Weak long-term credit assignment.",
    4:  "High LR + zero entropy. Fast but greedy policy collapse risk.",
    5:  "High entropy coefficient. Strong exploration, slow exploitation.",
    6:  "Doubled n_steps. More complete returns, lower bias.",
    7:  "Very low max_grad. Prevents large policy updates.",
    8:  "High entropy + lower gamma. Exploratory, moderate performance.",
    9:  "Very long rollouts (4 episodes). Reduces variance further.",
    10: "High LR + zero entropy. Likely unstable and divergent.",
}

# ═════════════════════════════════════════════════════════════════════════════
# PPO configurations
# ═════════════════════════════════════════════════════════════════════════════
PPO_CONFIGS = [
    {"run":1,  "lr":3e-4, "gamma":0.99, "n_steps":256,  "batch":64,  "n_epochs":10, "clip":0.2,  "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5},
    {"run":2,  "lr":1e-4, "gamma":0.99, "n_steps":512,  "batch":64,  "n_epochs":10, "clip":0.2,  "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5},
    {"run":3,  "lr":5e-4, "gamma":0.95, "n_steps":256,  "batch":128, "n_epochs":5,  "clip":0.2,  "ent_coef":0.00, "vf_coef":0.5,  "max_grad":0.5},
    {"run":4,  "lr":3e-4, "gamma":0.99, "n_steps":128,  "batch":32,  "n_epochs":20, "clip":0.1,  "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5},
    {"run":5,  "lr":3e-4, "gamma":0.99, "n_steps":256,  "batch":64,  "n_epochs":10, "clip":0.3,  "ent_coef":0.02, "vf_coef":0.5,  "max_grad":0.5},
    {"run":6,  "lr":1e-3, "gamma":0.99, "n_steps":1024, "batch":256, "n_epochs":5,  "clip":0.2,  "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5},
    {"run":7,  "lr":3e-4, "gamma":0.90, "n_steps":256,  "batch":64,  "n_epochs":10, "clip":0.2,  "ent_coef":0.01, "vf_coef":1.0,  "max_grad":0.5},
    {"run":8,  "lr":2e-4, "gamma":0.99, "n_steps":512,  "batch":128, "n_epochs":15, "clip":0.15, "ent_coef":0.05, "vf_coef":0.5,  "max_grad":1.0},
    {"run":9,  "lr":3e-4, "gamma":0.99, "n_steps":256,  "batch":64,  "n_epochs":10, "clip":0.2,  "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5},
    {"run":10, "lr":5e-3, "gamma":0.90, "n_steps":64,   "batch":32,  "n_epochs":3,  "clip":0.4,  "ent_coef":0.10, "vf_coef":0.1,  "max_grad":2.0},
]

PPO_NOTES = {
    1:  "Standard PPO baseline. Reliable convergence.",
    2:  "Lower LR + larger rollouts. More stable value estimates.",
    3:  "Higher LR + zero entropy. Deterministic policy faster.",
    4:  "More epochs + tighter clip. Conservative policy updates.",
    5:  "Wider clip range. Allows larger updates, may destabilise.",
    6:  "Large batch + high LR. Sample-efficient but noisy.",
    7:  "High vf_coef + lower gamma. Prioritises value accuracy.",
    8:  "High entropy + tight clip. Balances explore/exploit well.",
    9:  "Same as run 1 with different seed. Reproducibility check.",
    10: "Aggressive settings. High variance, likely sub-optimal.",
}

# ═════════════════════════════════════════════════════════════════════════════
# A2C configurations
# ═════════════════════════════════════════════════════════════════════════════
A2C_CONFIGS = [
    {"run":1,  "lr":7e-4, "gamma":0.99, "n_steps":5,   "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5,  "rms_eps":1e-5},
    {"run":2,  "lr":1e-3, "gamma":0.99, "n_steps":5,   "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5,  "rms_eps":1e-5},
    {"run":3,  "lr":5e-4, "gamma":0.95, "n_steps":10,  "ent_coef":0.01, "vf_coef":0.5,  "max_grad":0.5,  "rms_eps":1e-5},
    {"run":4,  "lr":7e-4, "gamma":0.99, "n_steps":5,   "ent_coef":0.05, "vf_coef":0.5,  "max_grad":0.5,  "rms_eps":1e-5},
    {"run":5,  "lr":7e-4, "gamma":0.99, "n_steps":5,   "ent_coef":0.00, "vf_coef":0.5,  "max_grad":0.5,  "rms_eps":1e-5},
    {"run":6,  "lr":2e-4, "gamma":0.99, "n_steps":20,  "ent_coef":0.01, "vf_coef":0.5,  "max_grad":1.0,  "rms_eps":1e-5},
    {"run":7,  "lr":7e-4, "gamma":0.99, "n_steps":5,   "ent_coef":0.01, "vf_coef":1.0,  "max_grad":0.5,  "rms_eps":1e-5},
    {"run":8,  "lr":1e-3, "gamma":0.90, "n_steps":5,   "ent_coef":0.01, "vf_coef":0.25, "max_grad":0.5,  "rms_eps":1e-4},
    {"run":9,  "lr":3e-4, "gamma":0.99, "n_steps":13,  "ent_coef":0.02, "vf_coef":0.5,  "max_grad":0.5,  "rms_eps":1e-5},
    {"run":10, "lr":5e-3, "gamma":0.90, "n_steps":5,   "ent_coef":0.10, "vf_coef":0.1,  "max_grad":2.0,  "rms_eps":1e-3},
]

A2C_NOTES = {
    1:  "Baseline A2C. Steady convergence, low variance.",
    2:  "Higher LR. Faster but slightly noisier learning.",
    3:  "Longer n_steps. Reduces bias in advantage estimates.",
    4:  "High entropy. More exploration, slower exploitation.",
    5:  "Zero entropy. Deterministic, risk of premature convergence.",
    6:  "Very long n_steps (20). Low-bias returns, high variance.",
    7:  "High vf_coef. Stronger value function signal.",
    8:  "Lower gamma + higher rms_eps. Less credit to future rewards.",
    9:  "Episode-length n_steps (13). Full-episode advantages.",
    10: "Aggressive LR + entropy. Likely unstable.",
}


# ── generic training function ─────────────────────────────────────────────────

def train_algorithm(algo_name: str, AlgoClass, configs: list, notes: dict, extra_kwargs_fn=None):
    algo_dir = os.path.join(BASE_DIR, algo_name)
    os.makedirs(algo_dir, exist_ok=True)
    results = []

    for cfg in configs:
        run_id   = cfg["run"]
        save_dir = os.path.join(algo_dir, f"run_{run_id}")
        os.makedirs(save_dir, exist_ok=True)

        train_env = make_vec_env(make_classroom_env(seed=run_id, n_students=N_STUDENTS), n_envs=4)
        eval_env  = make_vec_env(make_classroom_env(seed=run_id + 200, n_students=N_STUDENTS), n_envs=1)

        # build kwargs
        base_kwargs = dict(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg["lr"],
            gamma=cfg["gamma"],
            n_steps=cfg["n_steps"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad"],
            verbose=1,
            tensorboard_log=os.path.join(BASE_DIR, "tb_logs"),
            seed=run_id,
        )
        if extra_kwargs_fn:
            base_kwargs.update(extra_kwargs_fn(cfg))

        model = AlgoClass(**base_kwargs)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=save_dir,
            eval_freq=max(TOTAL_TIMESTEPS // 20, 2000),
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

        algo_display = algo_name.upper()
        print(f"\n{'='*65}")
        print(f"{algo_display} Run {run_id:>2} | LR={cfg['lr']:.0e} | "
              f"gamma={cfg['gamma']} | n_steps={cfg['n_steps']} | "
              f"ent={cfg['ent_coef']}")
        print(f"{'='*65}")

        t0 = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=True)
        elapsed = time.time() - t0

        model.save(os.path.join(save_dir, "final_model"))
        train_env.close()

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
        eval_env.close()

        print(f"  → Mean reward: {mean_r:.2f} ± {std_r:.2f}  (time: {elapsed:.0f}s)")

        results.append({
            "run": run_id,
            "lr": cfg["lr"],
            "gamma": cfg["gamma"],
            "n_steps": cfg["n_steps"],
            "ent_coef": cfg["ent_coef"],
            "vf_coef": cfg["vf_coef"],
            "max_grad_norm": cfg["max_grad"],
            "mean_reward": round(mean_r, 3),
            "std_reward": round(std_r, 3),
            "n_eval_episodes": 20,
            "training_time_s": round(elapsed, 1),
            "behaviour_notes": notes[run_id],
        })

    # save CSV
    csv_path = os.path.join(algo_dir, f"{algo_name}_results.csv")
    fields = ["run", "lr", "gamma", "n_steps", "ent_coef", "vf_coef",
              "max_grad_norm", "mean_reward", "std_reward",
              "n_eval_episodes", "training_time_s", "behaviour_notes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{algo_name.upper()} results saved → {csv_path}")
    _print_summary(algo_name.upper(), results)
    return results


def _print_summary(label, results):
    print(f"\n{'─'*80}")
    print(f"{label} Summary")
    print(f"{'─'*80}")
    print(f"{'Run':>4} {'LR':>8} {'γ':>5} {'n_steps':>8} "
          f"{'ent':>6} {'Mean R':>8} {'Std R':>7} {'Time(s)':>8}")
    for r in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
        print(f"{r['run']:>4} {r['lr']:>8.0e} {r['gamma']:>5} {r['n_steps']:>8} "
              f"{r['ent_coef']:>6} {r['mean_reward']:>8.3f} "
              f"{r['std_reward']:>7.3f} {r['training_time_s']:>8.1f}")


def ppo_extra(cfg):
    return {
        "batch_size": cfg["batch"],
        "n_epochs": cfg["n_epochs"],
        "clip_range": cfg["clip"],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    print("\n" + "█"*65)
    print("  REINFORCE (A2C, vf_coef=0, full-episode rollouts)")
    print("█"*65)
    reinforce_results = train_algorithm(
        algo_name="reinforce",
        AlgoClass=A2C,
        configs=REINFORCE_CONFIGS,
        notes=REINFORCE_NOTES,
    )
    all_results["reinforce"] = reinforce_results

    print("\n" + "█"*65)
    print("  PPO — Proximal Policy Optimization")
    print("█"*65)
    ppo_results = train_algorithm(
        algo_name="ppo",
        AlgoClass=PPO,
        configs=PPO_CONFIGS,
        notes=PPO_NOTES,
        extra_kwargs_fn=ppo_extra,
    )
    all_results["ppo"] = ppo_results

    print("\n" + "█"*65)
    print("  A2C — Advantage Actor-Critic")
    print("█"*65)
    a2c_results = train_algorithm(
        algo_name="a2c",
        AlgoClass=A2C,
        configs=A2C_CONFIGS,
        notes=A2C_NOTES,
    )
    all_results["a2c"] = a2c_results

    # ── find best overall PG model ────────────────────────────────────────────
    best_algo, best_run, best_reward = None, None, -np.inf
    for algo_name, results in all_results.items():
        for r in results:
            if r["mean_reward"] > best_reward:
                best_reward = r["mean_reward"]
                best_algo   = algo_name
                best_run    = r["run"]

    best_src  = os.path.join(BASE_DIR, best_algo, f"run_{best_run}")
    best_dest = os.path.join(BASE_DIR, "best_model")
    for fname in os.listdir(best_src):
        if "final_model" in fname or "best_model" in fname:
            shutil.copy(os.path.join(best_src, fname), best_dest)

    # write a small metadata file
    with open(os.path.join(best_dest, "metadata.txt"), "w") as f:
        f.write(f"algorithm={best_algo}\n")
        f.write(f"run={best_run}\n")
        f.write(f"mean_reward={best_reward:.3f}\n")

    print(f"\n{'='*65}")
    print(f"Best PG model: {best_algo.upper()} run #{best_run} | "
          f"Mean reward: {best_reward:.3f}")
    print(f"Saved → {best_dest}")


if __name__ == "__main__":
    main()
