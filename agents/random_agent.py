"""
Level 0: Random Agent
Establishes baseline performance by making stochastic decisions without state awareness.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


class RandomAgent:
    """
    Baseline agent that flaps with a fixed Bernoulli probability `flap_prob`.
    Default `flap_prob = 0.12` roughly balances gravity so the bird hovers randomly
    instead of flying straight into the ceiling on frame 15.
    """

    level = 0
    name = "Random Agent"

    def __init__(self, flap_prob: float = 0.12, seed: Optional[int] = None):
        self.flap_prob = flap_prob
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        return 1 if self.rng.random() < self.flap_prob else 0

    def get_decision_info(self, state: np.ndarray) -> Dict[str, float]:
        return {
            "q_no_flap": 1.0 - self.flap_prob,
            "q_flap": self.flap_prob,
        }
