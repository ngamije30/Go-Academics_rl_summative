"""
GoAcademics - Main Entry Point
--------------------------------
Loads and runs the best-performing trained model against the GoAcademicsEnv
classroom simulation with full pygame GUI and terminal verbose output.

Usage:
    python main.py                          # auto-detect best model
    python main.py --algo dqn               # run best DQN model
    python main.py --algo ppo               # run best PPO model
    python main.py --algo a2c               # run best A2C model
    python main.py --algo reinforce         # run best REINFORCE model
    python main.py --episodes 3             # run N episodes
    python main.py --students 15            # classroom size
    python main.py --no-render              # terminal only (no GUI)
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTION_NAMES = {
    0: "No Action",
    1: "Parent Alert",
    2: "Peer Tutor",
    3: "Counselling",
    4: "Extra CA",
    5: "Escalate",
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def find_best_model(algo: str):
    """
    Locate the best saved model for a given algorithm.
    Priority: <algo>/best_model/ → <algo>/run_N/best_model.zip → run_N/final_model.zip
    """
    if algo == "dqn":
        algo_dir = os.path.join(MODELS_DIR, "dqn")
    else:
        algo_dir = os.path.join(MODELS_DIR, "pg", algo)

    # 1) dedicated best_model folder
    if algo == "dqn":
        best_dir = os.path.join(MODELS_DIR, "dqn", "best_model")
    else:
        best_dir = os.path.join(MODELS_DIR, "pg", "best_model")

    for candidate in ["best_model.zip", "final_model.zip"]:
        path = os.path.join(best_dir, candidate)
        if os.path.exists(path):
            return path.replace(".zip", "")

    # 2) scan runs, pick highest numbered best_model.zip
    if os.path.exists(algo_dir):
        run_dirs = sorted(
            [d for d in os.listdir(algo_dir) if d.startswith("run_")],
            key=lambda x: int(x.split("_")[1])
        )
        for run_dir in reversed(run_dirs):
            for candidate in ["best_model.zip", "final_model.zip"]:
                path = os.path.join(algo_dir, run_dir, candidate)
                if os.path.exists(path):
                    return path.replace(".zip", "")

    return None


def load_model(algo: str, model_path: str):
    """Load SB3 model for the given algorithm."""
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    elif algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    elif algo in ("a2c", "reinforce"):
        from stable_baselines3 import A2C
        return A2C.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def detect_best_overall() -> tuple[str, str]:
    """Read metadata files to find the globally best model."""
    pg_meta = os.path.join(MODELS_DIR, "pg", "best_model", "metadata.txt")
    if os.path.exists(pg_meta):
        meta = {}
        with open(pg_meta) as f:
            for line in f:
                k, v = line.strip().split("=")
                meta[k] = v
        pg_algo    = meta.get("algorithm", "ppo")
        pg_reward  = float(meta.get("mean_reward", -999))
        pg_path    = find_best_model(pg_algo)
    else:
        pg_algo, pg_reward, pg_path = "ppo", -999, None

    dqn_path  = find_best_model("dqn")
    dqn_reward = -999
    if dqn_path:
        results_csv = os.path.join(MODELS_DIR, "dqn", "dqn_results.csv")
        if os.path.exists(results_csv):
            import csv
            with open(results_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                best_row = max(rows, key=lambda r: float(r["mean_reward"]))
                dqn_reward = float(best_row["mean_reward"])

    if dqn_reward >= pg_reward and dqn_path:
        return "dqn", dqn_path
    elif pg_path:
        return pg_algo, pg_path
    else:
        raise ValueError("No trained models found. Run training scripts first.")


def run_simulation(algo: str, model, n_episodes: int, n_students: int, render: bool):
    """Run the full simulation with verbose terminal output and optional GUI."""
    from environment.custom_env import GoAcademicsEnv, SingleStudentEnv

    use_classroom = algo in ("ppo", "a2c", "reinforce")

    if use_classroom:
        env = GoAcademicsEnv(
            n_students=n_students,
            render_mode="human" if render else None,
        )
    else:
        # DQN uses single-student env; we manually loop over a classroom
        env = SingleStudentEnv()

    print("\n" + "="*70)
    print("  GoAcademics — Early Warning System  |  RL Agent Demo")
    print("="*70)
    print(f"  Problem  : Predict & intervene for at-risk secondary students")
    print(f"  Algorithm: {algo.upper()}")
    print(f"  Students : {n_students}")
    print(f"  Objective: Maximise end-of-term pass rate via timely interventions")
    print(f"  Rewards  : +10 risk-drop | +20 pass | -20 dropout | -5 risk-rise")
    print("="*70)

    if render:
        try:
            import pygame
            pygame.init()
        except ImportError:
            print("  [WARNING] pygame not installed. Running without GUI.")
            render = False

    episode_rewards = []

    for ep in range(n_episodes):
        if use_classroom:
            obs, info = env.reset()
        else:
            obs, info = env.reset()

        done        = False
        ep_reward   = 0.0
        step        = 0
        n_passed    = 0
        n_dropped   = 0

        print(f"\n── Episode {ep + 1}/{n_episodes} ──────────────────────────────")

        while not done:
            # handle pygame quit
            if render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        sys.exit()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

            # terminal verbose
            if use_classroom:
                week      = info.get("term_week", step)
                n_passed  = info.get("n_passed", 0)
                n_dropped = info.get("n_dropped", 0)
                risk      = info.get("risk_counts", {})
                avg_ca    = info.get("avg_ca", 0)
                avg_att   = info.get("avg_attendance", 0)

                action_list = list(action) if hasattr(action, "__iter__") else [action]
                action_names = [ACTION_NAMES.get(int(a), str(a)) for a in action_list]
                unique_actions = list(dict.fromkeys(action_names))

                print(
                    f"  Wk {week:>2} | "
                    f"Reward: {reward:>+7.1f} | "
                    f"Passed: {n_passed:>2}/{n_students} | "
                    f"Dropped: {n_dropped} | "
                    f"High-Risk: {risk.get('high', 0)} | "
                    f"Avg CA: {avg_ca:>5.1f} | "
                    f"Avg Att: {avg_att:.0%} | "
                    f"Actions: {unique_actions}"
                )
            else:
                ca    = info.get("ca_score", 0)
                att   = info.get("attendance", 0)
                risk  = info.get("risk_level", 0)
                act_n = ACTION_NAMES.get(int(action), str(action))
                print(
                    f"  Wk {step:>2} | "
                    f"Reward: {reward:>+7.1f} | "
                    f"CA: {ca:>5.1f} | "
                    f"Att: {att:.0%} | "
                    f"Risk: {risk} | "
                    f"Action: {act_n}"
                )

        episode_rewards.append(ep_reward)
        print(f"\n  Episode {ep + 1} Summary:")
        print(f"    Total reward : {ep_reward:.2f}")
        print(f"    Steps        : {step}")
        if use_classroom:
            print(f"    Passed       : {n_passed}/{n_students} students")
            print(f"    Dropped out  : {n_dropped} students")

    if render:
        try:
            import pygame
            print("\n  [GUI] Simulation ended. Close the window or press any key to exit.")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type in (pygame.QUIT, pygame.KEYDOWN):
                        waiting = False
                env.clock.tick(10)
        except Exception:
            pass

    env.close()
    if render:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    print("\n" + "="*70)
    print(f"  Simulation complete — {n_episodes} episode(s)")
    print(f"  Mean episode reward : {np.mean(episode_rewards):.2f}")
    print(f"  Std                 : {np.std(episode_rewards):.2f}")
    print(f"  Best episode        : {np.max(episode_rewards):.2f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="GoAcademics RL Demo")
    parser.add_argument("--algo",     type=str, default="auto",
                        choices=["auto", "dqn", "ppo", "a2c", "reinforce"],
                        help="Algorithm to run (default: auto-detect best)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to simulate (default: 3)")
    parser.add_argument("--students", type=int, default=10,
                        help="Number of students in classroom (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable pygame GUI (terminal only)")
    args = parser.parse_args()

    render = not args.no_render

    # ── resolve algorithm + model path ────────────────────────────────────────
    if args.algo == "auto":
        algo, model_path = detect_best_overall()
        print(f"  [Auto-detect] Best algorithm: {algo.upper()}")
    else:
        algo = args.algo
        model_path = find_best_model(algo)

    if model_path is None or not (
        os.path.exists(model_path + ".zip") or os.path.exists(model_path)
    ):
        print(f"\n[ERROR] No trained model found for '{algo}'.")
        print("Please run the training scripts first:")
        print("    python training/dqn_training.py")
        print("    python training/pg_training.py")
        sys.exit(1)

    print(f"\n  Loading model: {model_path}")
    model = load_model(algo, model_path)

    run_simulation(
        algo=algo,
        model=model,
        n_episodes=args.episodes,
        n_students=args.students,
        render=render,
    )


if __name__ == "__main__":
    main()
