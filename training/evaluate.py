"""
Multi-seed evaluation utility for FlappyRL agents.
Evaluates any agent with exploration disabled across unseen random seeds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol
import numpy as np

from environment.flappy_bird import FlappyBirdEnv


class SupportsSelectAction(Protocol):
    def select_action(self, state: np.ndarray, training: bool = False) -> int: ...


def evaluate_agent(
    agent: SupportsSelectAction,
    seeds: List[int] | None = None,
    max_steps: int = 3000,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

    scores: List[int] = []
    rewards: List[float] = []
    steps_list: List[int] = []

    for seed in seeds:
        env = FlappyBirdEnv(seed=seed, reward_shaping=False, max_steps=max_steps)
        state = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        scores.append(env.score)
        rewards.append(round(env.cumulative_reward, 2))
        steps_list.append(env.steps)

    scores_arr = np.array(scores, dtype=np.float64)
    rewards_arr = np.array(rewards, dtype=np.float64)
    steps_arr = np.array(steps_list, dtype=np.float64)

    return {
        "mean_score": round(float(scores_arr.mean()), 2),
        "std_score": round(float(scores_arr.std()), 2),
        "max_score": int(scores_arr.max()),
        "min_score": int(scores_arr.min()),
        "mean_reward": round(float(rewards_arr.mean()), 2),
        "std_reward": round(float(rewards_arr.std()), 2),
        "mean_steps": round(float(steps_arr.mean()), 1),
        "per_seed_scores": scores,
    }
