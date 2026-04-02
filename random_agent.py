"""
GoAcademics - Random Agent Demo
---------------------------------
Runs the GoAcademics classroom environment with a fully random agent
(no trained model) and renders every step using the pygame GUI.

This file demonstrates the environment's visualization and components
without any training involved — satisfying the rubric requirement for
a static file showing the agent taking random actions.

Usage:
    python random_agent.py                   # 10 students, 2 episodes
    python random_agent.py --students 15     # custom classroom size
    python random_agent.py --episodes 3      # more episodes
    python random_agent.py --no-render       # terminal only (no GUI)
    python random_agent.py --seed 42         # fixed seed for reproducibility
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


def run_random_agent(n_students: int = 10, n_episodes: int = 2,
                     render: bool = True, seed: int = None):
    from environment.custom_env import GoAcademicsEnv

    env = GoAcademicsEnv(
        n_students=n_students,
        render_mode="human" if render else None,
        seed=seed,
    )

    if render:
        try:
            import pygame
            pygame.init()
        except ImportError:
            print("[WARNING] pygame not installed. Running without GUI.")
            render = False
            env.render_mode = None

    print("\n" + "=" * 65)
    print("  GoAcademics — Random Agent Demo")
    print("=" * 65)
    print(f"  Environment : GoAcademicsEnv ({n_students} students)")
    print(f"  Agent       : Random (no model — uniform action sampling)")
    print(f"  Purpose     : Visualise environment components pre-training")
    print(f"  Actions     : {list(ACTION_NAMES.values())}")
    print(f"  Observation : 6 features per student (attendance, CA score,")
    print(f"                study hours, assignment completion,")
    print(f"                term week, risk level)")
    print(f"  Rewards     : +10 risk-drop | +20 pass | -20 dropout |")
    print(f"                -5 risk-rise | -1 ignore high-risk |")
    print(f"                -2 unnecessary escalation")
    print(f"  Terminal    : End of 13-week term OR all students resolved")
    print("=" * 65)

    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        step = 0

        print(f"\n── Episode {ep + 1}/{n_episodes} "
              f"({'Random Actions — No Policy' }) ──────────────")

        while not done:
            if render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        sys.exit()

            # random action — sample uniformly from action space
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

            week      = info.get("term_week", step)
            n_passed  = info.get("n_passed", 0)
            n_dropped = info.get("n_dropped", 0)
            risk      = info.get("risk_counts", {})
            avg_ca    = info.get("avg_ca", 0.0)
            avg_att   = info.get("avg_attendance", 0.0)

            action_names = [ACTION_NAMES.get(int(a), str(a)) for a in action]
            unique = list(dict.fromkeys(action_names))

            print(
                f"  Wk {week:>2} | "
                f"Reward: {reward:>+8.1f} | "
                f"Passed: {n_passed:>2}/{n_students} | "
                f"Dropped: {n_dropped} | "
                f"High-Risk: {risk.get('high', 0)} | "
                f"Avg CA: {avg_ca:>5.1f} | "
                f"Avg Att: {avg_att:.0%} | "
                f"Actions: {unique}"
            )

        episode_rewards.append(ep_reward)
        print(f"\n  Episode {ep + 1} Summary:")
        print(f"    Total reward : {ep_reward:.2f}")
        print(f"    Steps        : {step}")
        print(f"    Passed       : {info.get('n_passed', 0)}/{n_students}")
        print(f"    Dropped out  : {info.get('n_dropped', 0)}")

    env.close()
    if render:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(f"  Random Agent — {n_episodes} episode(s) complete")
    print(f"  Mean reward : {np.mean(episode_rewards):.2f}")
    print(f"  Std         : {np.std(episode_rewards):.2f}")
    print(f"  (Compare this baseline against trained agents in main.py)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="GoAcademics Random Agent Demo — no trained model"
    )
    parser.add_argument("--students",  type=int, default=10,
                        help="Number of students in classroom (default: 10)")
    parser.add_argument("--episodes",  type=int, default=2,
                        help="Number of episodes to run (default: 2)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable pygame GUI (terminal output only)")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    run_random_agent(
        n_students=args.students,
        n_episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
