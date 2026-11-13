"""
Gymnasium-compatible wrapper around FlappyBirdEnv.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from environment.flappy_bird import FlappyBirdEnv


class FlappyBirdGymEnv(gym.Env):
    """
    Standard Gymnasium environment wrapping FlappyBirdEnv.
    Observation space: Box(5,) float32
    Action space: Discrete(2) (0: do nothing, 1: flap)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        seed: Optional[int] = None,
        reward_shaping: bool = False,
        max_steps: int = 10000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.core = FlappyBirdEnv(
            seed=seed, reward_shaping=reward_shaping, max_steps=max_steps
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -0.5, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.5, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs = self.core.reset(seed=seed)
        return obs, self.core.get_raw_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.core.step(int(action))
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            # Lightweight synthetic frame representation if needed by tests
            frame = np.zeros((600, 400, 3), dtype=np.uint8)
            return frame
        return None
