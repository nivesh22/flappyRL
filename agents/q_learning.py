"""
Level 2: Tabular Q-Learning Agent
Discretizes key relative state variables into compact buckets and learns
a Bellman Q-table Q(s_discrete, a).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class TabularQLearningAgent:
    """
    Discretizes continuous state into bins over:
      - dy_center = (dy_lower - dy_upper): relative vertical offset from gap center
      - bird_vel: vertical velocity
      - dx: horizontal distance to pipe
    """

    level = 2
    name = "Tabular Q-Learning"

    def __init__(
        self,
        dy_bins: int = 14,
        vel_bins: int = 8,
        dx_bins: int = 6,
        lr: float = 0.15,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.992,
        seed: Optional[int] = 42,
    ):
        self.dy_bins = dy_bins
        self.vel_bins = vel_bins
        self.dx_bins = dx_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.dy_edges = np.linspace(-0.02, 0.22, dy_bins - 1)
        self.vel_edges = np.linspace(-0.65, 0.75, vel_bins - 1)
        self.dx_edges = np.linspace(0.0, 0.8, dx_bins - 1)

        # Initialize Q-table with slight bias toward not flapping (0) to prevent ceiling crash early
        self.q_table = np.zeros((dy_bins, vel_bins, dx_bins, 2), dtype=np.float32)
        self.q_table[..., 0] = 0.05

    def discretize(self, state: np.ndarray) -> Tuple[int, int, int]:
        bird_vel = float(state[1])
        dx = float(state[2])
        dy_lower = float(state[4])

        b_dy = int(np.digitize(dy_lower, self.dy_edges))
        b_vel = int(np.digitize(bird_vel, self.vel_edges))
        b_dx = int(np.digitize(dx, self.dx_edges))
        return b_dy, b_vel, b_dx

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        idx = self.discretize(state)
        return self.q_table[idx]

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and self.rng.random() < self.epsilon:
            return 1 if self.rng.random() < 0.18 else 0
        q_vals = self.get_q_values(state)
        return int(np.argmax(q_vals))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        s_idx = self.discretize(state)
        ns_idx = self.discretize(next_state)

        current_q = self.q_table[s_idx][action]
        target_q = (
            reward
            if done
            else reward + self.gamma * float(np.max(self.q_table[ns_idx]))
        )
        td_error = target_q - current_q
        self.q_table[s_idx][action] += self.lr * td_error
        return float(abs(td_error))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dy_bins": self.dy_bins,
            "vel_bins": self.vel_bins,
            "dx_bins": self.dx_bins,
            "dy_edges": self.dy_edges.tolist(),
            "vel_edges": self.vel_edges.tolist(),
            "dx_edges": self.dx_edges.tolist(),
            "q_table": self.q_table.tolist(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TabularQLearningAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            dy_bins=data["dy_bins"],
            vel_bins=data["vel_bins"],
            dx_bins=data["dx_bins"],
        )
        agent.q_table = np.array(data["q_table"], dtype=np.float32)
        agent.epsilon = 0.0
        return agent
