"""
Core Flappy Bird physics engine for Reinforcement Learning.
Designed to be 100% deterministic with a portable Mulberry32 PRNG so that
Python training/evaluation and TypeScript HTML5 Canvas produce identical
pipe sequences for any given seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class GameConstants:
    WIDTH: float = 400.0
    HEIGHT: float = 600.0
    GROUND_Y: float = 520.0
    BIRD_X: float = 80.0
    BIRD_RADIUS: float = 14.0
    GRAVITY: float = 0.45
    FLAP_STRENGTH: float = -7.5
    MAX_VEL_Y: float = 10.0
    MIN_VEL_Y: float = -9.0
    PIPE_WIDTH: float = 56.0
    PIPE_GAP: float = 120.0
    PIPE_SPEED: float = 3.0
    PIPE_SPAWN_SPACING: float = 210.0
    FIRST_PIPE_X: float = 320.0
    GAP_MIN_Y: float = 80.0
    GAP_MAX_Y: float = 310.0


class Mulberry32:
    """
    Portable 32-bit deterministic PRNG matching the exact bitwise implementation
    in the TypeScript frontend (`frontend/src/engine/prng.ts`).
    """

    def __init__(self, seed: int = 42):
        self.state = seed & 0xFFFFFFFF

    def next_float(self) -> float:
        """Returns a float in [0.0, 1.0)."""
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) ^ t
        t = (t ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296.0

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.next_float()


@dataclass
class Pipe:
    x: float
    gap_top_y: float
    gap_bottom_y: float
    passed: bool = False

    @property
    def gap_center_y(self) -> float:
        return 0.5 * (self.gap_top_y + self.gap_bottom_y)


class FlappyBirdEnv:
    """
    Pure numerical Flappy Bird environment following the Project Specification:
    State space (5D normalized float32):
      0: bird_y / GROUND_Y
      1: bird_vel / MAX_VEL_Y
      2: horizontal distance to next pipe right edge / WIDTH
      3: vertical distance to upper pipe lip: (bird_y - gap_top_y) / GROUND_Y
      4: vertical distance to lower pipe lip: (gap_bottom_y - bird_y) / GROUND_Y

    Action space:
      0: Do nothing
      1: Flap wings

    Reward function:
      Survive one timestep: +0.1
      Successfully pass a pipe: +1.0
      Collision or death: -10.0
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        reward_shaping: bool = False,
        max_steps: int = 10000,
    ):
        self.c = GameConstants()
        self.base_seed = seed if seed is not None else 42
        self.prng = Mulberry32(self.base_seed)
        self.reward_shaping = reward_shaping
        self.max_steps = max_steps

        self.bird_y: float = self.c.GROUND_Y * 0.45
        self.bird_vel: float = 0.0
        self.pipes: List[Pipe] = []
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = False
        self.cumulative_reward: float = 0.0

        self.reset(seed=self.base_seed)

    def _spawn_pipe(self, x: float) -> Pipe:
        gap_top = round(self.prng.uniform(self.c.GAP_MIN_Y, self.c.GAP_MAX_Y), 2)
        return Pipe(
            x=x,
            gap_top_y=gap_top,
            gap_bottom_y=gap_top + self.c.PIPE_GAP,
            passed=False,
        )

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.base_seed = int(seed)
        self.prng = Mulberry32(self.base_seed)

        self.bird_y = self.c.GROUND_Y * 0.45  # 234.0
        self.bird_vel = 0.0
        self.score = 0
        self.steps = 0
        self.done = False
        self.cumulative_reward = 0.0

        self.pipes = [
            self._spawn_pipe(self.c.FIRST_PIPE_X),
            self._spawn_pipe(self.c.FIRST_PIPE_X + self.c.PIPE_SPAWN_SPACING),
            self._spawn_pipe(self.c.FIRST_PIPE_X + 2.0 * self.c.PIPE_SPAWN_SPACING),
        ]
        return self.get_state()

    def get_next_pipe(self) -> Pipe:
        """Returns the first pipe whose right edge is still ahead of the bird's back edge."""
        bird_left = self.c.BIRD_X - self.c.BIRD_RADIUS
        for pipe in self.pipes:
            if pipe.x + self.c.PIPE_WIDTH >= bird_left:
                return pipe
        return self.pipes[0]

    def get_state(self) -> np.ndarray:
        next_pipe = self.get_next_pipe()
        dx = (next_pipe.x + self.c.PIPE_WIDTH - self.c.BIRD_X) / self.c.WIDTH
        dy_upper = (self.bird_y - next_pipe.gap_top_y) / self.c.GROUND_Y
        dy_lower = (next_pipe.gap_bottom_y - self.bird_y) / self.c.GROUND_Y

        return np.array(
            [
                self.bird_y / self.c.GROUND_Y,
                self.bird_vel / self.c.MAX_VEL_Y,
                dx,
                dy_upper,
                dy_lower,
            ],
            dtype=np.float32,
        )

    def get_raw_info(self) -> Dict[str, Any]:
        next_pipe = self.get_next_pipe()
        return {
            "bird_y": self.bird_y,
            "bird_vel": self.bird_vel,
            "next_pipe_x": next_pipe.x,
            "next_pipe_gap_top": next_pipe.gap_top_y,
            "next_pipe_gap_bottom": next_pipe.gap_bottom_y,
            "next_pipe_gap_center": next_pipe.gap_center_y,
            "score": self.score,
            "steps": self.steps,
            "cumulative_reward": self.cumulative_reward,
        }

    def _check_collision(self) -> bool:
        # Hit ceiling or ground
        if (
            self.bird_y - self.c.BIRD_RADIUS <= 0.0
            or self.bird_y + self.c.BIRD_RADIUS >= self.c.GROUND_Y
        ):
            return True

        # Check pipes using circle-AABB intersection (or tight bounding check)
        bx = self.c.BIRD_X
        by = self.bird_y
        r = self.c.BIRD_RADIUS - 2.0  # slight forgiveness matching standard Flappy Bird feel

        for pipe in self.pipes:
            if bx + r > pipe.x and bx - r < pipe.x + self.c.PIPE_WIDTH:
                if by - r < pipe.gap_top_y or by + r > pipe.gap_bottom_y:
                    return True
        return False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.done:
            return self.get_state(), 0.0, True, False, self.get_raw_info()

        # 1. Apply action
        if int(action) == 1:
            self.bird_vel = self.c.FLAP_STRENGTH
        else:
            self.bird_vel = min(self.c.MAX_VEL_Y, self.bird_vel + self.c.GRAVITY)

        # 2. Update bird position
        self.bird_y += self.bird_vel
        self.steps += 1

        # 3. Update pipes and check pipe passing
        passed_pipe = False
        for pipe in self.pipes:
            pipe.x -= self.c.PIPE_SPEED
            if not pipe.passed and (pipe.x + self.c.PIPE_WIDTH) < self.c.BIRD_X:
                pipe.passed = True
                self.score += 1
                passed_pipe = True

        # Remove off-screen pipes and spawn new ones
        if self.pipes and self.pipes[0].x + self.c.PIPE_WIDTH < -10.0:
            self.pipes.pop(0)
            rightmost_x = max(p.x for p in self.pipes)
            self.pipes.append(self._spawn_pipe(rightmost_x + self.c.PIPE_SPAWN_SPACING))

        # 4. Check collision
        collision = self._check_collision()
        truncated = self.steps >= self.max_steps and not collision
        self.done = collision or truncated

        # 5. Compute canonical reward
        if collision:
            reward = -10.0
        else:
            reward = 0.1
            if passed_pipe:
                reward += 1.0
            if self.reward_shaping:
                next_pipe = self.get_next_pipe()
                dist_from_center = abs(self.bird_y - next_pipe.gap_center_y) / (
                    self.c.GROUND_Y * 0.5
                )
                reward += 0.05 * max(0.0, 1.0 - dist_from_center)

        self.cumulative_reward += reward
        return self.get_state(), float(reward), collision, truncated, self.get_raw_info()
