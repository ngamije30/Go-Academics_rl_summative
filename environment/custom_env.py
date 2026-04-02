"""
GoAcademics - Custom Multi-Agent Gymnasium Environment
-------------------------------------------------------
Each 'agent' represents a student in a Rwandan secondary school classroom.
The RL agent (teacher/system) observes all students and selects one
intervention per student per timestep to maximise end-of-term pass rates.

Observation per student (6 features):
    [attendance_rate, ca_score, study_hours,
     assignment_completion, term_week, risk_level]

Actions (discrete, 6 per student):
    0 - No intervention
    1 - Send attendance alert to parent
    2 - Assign peer tutoring
    3 - One-on-one teacher counselling
    4 - Extra CA practice assignment
    5 - Escalate to school administration

Terminal conditions:
    - End of term (week 13)
    - All students have passed (ca >= 50 AND attendance >= 0.70)
    - A student drops out (attendance < 0.20 for 3 consecutive weeks)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── constants ────────────────────────────────────────────────────────────────
N_ACTIONS = 6
TERM_LENGTH = 13          # weeks
PASS_CA = 50.0
PASS_ATTENDANCE = 0.70
DROPOUT_ATTENDANCE = 0.20
DROPOUT_WEEKS = 3         # consecutive low-attendance weeks → dropout


class StudentState:
    """Holds the mutable state for one student."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.reset()

    def reset(self):
        self.attendance_rate: float = float(self.rng.uniform(0.50, 1.00))
        self.ca_score: float = float(self.rng.uniform(20.0, 80.0))
        self.study_hours: float = float(self.rng.uniform(0.5, 6.0))
        self.assignment_completion: float = float(self.rng.uniform(0.40, 1.00))
        self.risk_level: int = self._compute_risk()
        self.low_attendance_streak: int = 0
        self.dropped_out: bool = False
        self.passed: bool = False

    # ── derived ──────────────────────────────────────────────────────────────
    def _compute_risk(self) -> int:
        score = 0
        if self.attendance_rate < 0.60:
            score += 1
        if self.ca_score < 40:
            score += 1
        if self.study_hours < 2.0:
            score += 1
        if self.assignment_completion < 0.60:
            score += 1
        if score >= 3:
            return 2   # high
        elif score >= 1:
            return 1   # medium
        return 0       # low

    def to_obs(self, term_week: int) -> np.ndarray:
        return np.array([
            self.attendance_rate,
            self.ca_score / 100.0,            # normalise to [0,1]
            self.study_hours / 12.0,
            self.assignment_completion,
            term_week / TERM_LENGTH,
            self.risk_level / 2.0,
        ], dtype=np.float32)

    # ── transition ───────────────────────────────────────────────────────────
    def step(self, action: int, rng: np.random.Generator) -> float:
        """Apply one intervention, update state, return reward."""
        old_risk = self.risk_level
        reward = 0.0

        # ── apply action effects ──────────────────────────────────────────
        if action == 0:   # no intervention
            if self.risk_level == 2:
                reward -= 1.0   # penalise ignoring high-risk student
            # natural drift: slight random change
            self.ca_score += float(rng.uniform(-1.0, 1.5))
            self.attendance_rate += float(rng.uniform(-0.02, 0.01))

        elif action == 1:  # attendance alert to parent
            self.attendance_rate += float(rng.uniform(0.02, 0.08))
            self.ca_score += float(rng.uniform(-0.5, 1.0))

        elif action == 2:  # peer tutoring
            self.ca_score += float(rng.uniform(1.0, 4.0))
            self.study_hours += float(rng.uniform(0.2, 0.8))

        elif action == 3:  # one-on-one counselling
            self.ca_score += float(rng.uniform(1.5, 5.0))
            self.attendance_rate += float(rng.uniform(0.01, 0.05))
            self.study_hours += float(rng.uniform(0.1, 0.5))

        elif action == 4:  # extra CA practice
            self.ca_score += float(rng.uniform(2.0, 6.0))
            self.assignment_completion += float(rng.uniform(0.02, 0.08))

        elif action == 5:  # escalate to administration
            if self.risk_level < 2:
                reward -= 2.0   # penalise unnecessary escalation
            else:
                self.attendance_rate += float(rng.uniform(0.03, 0.10))
                self.ca_score += float(rng.uniform(1.0, 4.0))
                self.study_hours += float(rng.uniform(0.2, 0.6))

        # ── clamp values ─────────────────────────────────────────────────
        self.ca_score = float(np.clip(self.ca_score, 0.0, 100.0))
        self.attendance_rate = float(np.clip(self.attendance_rate, 0.0, 1.0))
        self.study_hours = float(np.clip(self.study_hours, 0.0, 12.0))
        self.assignment_completion = float(np.clip(self.assignment_completion, 0.0, 1.0))

        # ── recompute risk ────────────────────────────────────────────────
        new_risk = self._compute_risk()
        self.risk_level = new_risk

        # ── risk-change rewards ───────────────────────────────────────────
        if new_risk < old_risk:
            reward += 10.0
        elif new_risk > old_risk:
            reward -= 5.0

        # ── CA improvement ────────────────────────────────────────────────
        if self.ca_score > (PASS_CA - 5):
            reward += 2.0

        # ── dropout tracking ─────────────────────────────────────────────
        if self.attendance_rate < DROPOUT_ATTENDANCE:
            self.low_attendance_streak += 1
        else:
            self.low_attendance_streak = 0

        if self.low_attendance_streak >= DROPOUT_WEEKS:
            self.dropped_out = True
            reward -= 20.0

        # ── pass check ───────────────────────────────────────────────────
        if self.ca_score >= PASS_CA and self.attendance_rate >= PASS_ATTENDANCE:
            self.passed = True
            reward += 20.0

        return reward


# ── main environment ─────────────────────────────────────────────────────────

class GoAcademicsEnv(gym.Env):
    """
    Multi-student classroom environment.

    Observation  : Box(n_students * 6,)  — flattened student feature vectors
    Action       : MultiDiscrete([N_ACTIONS] * n_students)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, n_students: int = 10, render_mode: str = None, seed: int = None):
        super().__init__()

        self.n_students = n_students
        self.render_mode = render_mode
        self._seed = seed

        self.rng = np.random.default_rng(seed)

        # spaces
        obs_low = np.zeros(n_students * 6, dtype=np.float32)
        obs_high = np.ones(n_students * 6, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * n_students)

        # students
        self.students = [StudentState(np.random.default_rng(i)) for i in range(n_students)]
        self.term_week = 1

        # rendering
        self.window = None
        self.clock = None
        self.window_size = (1100, 600)

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        for s in self.students:
            s.rng = np.random.default_rng(self.rng.integers(0, 2**31))
            s.reset()

        self.term_week = 1
        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, actions):
        assert len(actions) == self.n_students, "Must provide one action per student"

        total_reward = 0.0
        for i, (student, action) in enumerate(zip(self.students, actions)):
            if not student.dropped_out:
                total_reward += student.step(int(action), self.rng)

        self.term_week += 1

        terminated = self._check_terminated()
        truncated = self.term_week > TERM_LENGTH
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, total_reward, terminated, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([s.to_obs(self.term_week) for s in self.students])
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "term_week": self.term_week,
            "n_passed": sum(s.passed for s in self.students),
            "n_dropped": sum(s.dropped_out for s in self.students),
            "risk_counts": {
                "low": sum(1 for s in self.students if s.risk_level == 0),
                "medium": sum(1 for s in self.students if s.risk_level == 1),
                "high": sum(1 for s in self.students if s.risk_level == 2),
            },
            "avg_ca": float(np.mean([s.ca_score for s in self.students])),
            "avg_attendance": float(np.mean([s.attendance_rate for s in self.students])),
        }

    def _check_terminated(self) -> bool:
        all_resolved = all(s.passed or s.dropped_out for s in self.students)
        return all_resolved

    # ── rendering ─────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        try:
            from environment.rendering import render_classroom
        except ImportError:
            from rendering import render_classroom
        return render_classroom(self)

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None


# ── single-student wrapper (for DQN / VecEnv) ────────────────────────────────

class SingleStudentEnv(gym.Env):
    """
    Wraps one student for use with algorithms that require Discrete actions
    (e.g. DQN). Compatible with SB3 VecEnv.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, seed: int = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.student = StudentState(np.random.default_rng(self.rng.integers(0, 2**31)))
        self.term_week = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.student.rng = np.random.default_rng(self.rng.integers(0, 2**31))
        self.student.reset()
        self.term_week = 1
        return self.student.to_obs(self.term_week), {}

    def step(self, action):
        reward = self.student.step(int(action), self.rng)
        self.term_week += 1
        terminated = self.student.passed or self.student.dropped_out
        truncated = self.term_week > TERM_LENGTH
        obs = self.student.to_obs(self.term_week)
        info = {
            "term_week": self.term_week,
            "ca_score": self.student.ca_score,
            "attendance": self.student.attendance_rate,
            "risk_level": self.student.risk_level,
            "passed": self.student.passed,
            "dropped_out": self.student.dropped_out,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
