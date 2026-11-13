"""
Level 1: Rule-Based Heuristic Agent
Establishes a non-ML benchmark using domain-specific threshold rules on the 5D state.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


class RuleBasedAgent:
    """
    Hand-crafted threshold heuristic:
    State indices:
      0: bird_y / GROUND_Y
      1: bird_vel / MAX_VEL_Y
      2: dx / WIDTH
      3: dy_upper = (bird_y - gap_top_y) / GROUND_Y
      4: dy_lower = (gap_bottom_y - bird_y) / GROUND_Y

    Flaps when the bird drops into the lower portion of the upcoming pipe gap
    and isn't already rising fast.
    """

    level = 1
    name = "Rule-Based Agent"

    def __init__(self, margin: float = 0.065, vel_threshold: float = -0.15):
        self.margin = margin
        self.vel_threshold = vel_threshold

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        bird_vel = float(state[1])
        dy_lower = float(state[4])
        # If close to or below the lower pipe lip (minus safety margin) and not rising strongly, flap
        if dy_lower < self.margin and bird_vel >= self.vel_threshold:
            return 1
        return 0

    def get_decision_info(self, state: np.ndarray) -> Dict[str, float]:
        bird_vel = float(state[1])
        dy_lower = float(state[4])
        urgency = (self.margin - dy_lower) * 10.0 + bird_vel
        return {
            "q_no_flap": 0.0,
            "q_flap": float(urgency),
        }
