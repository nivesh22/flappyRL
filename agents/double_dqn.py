"""
Level 5: Double Deep Q-Network (Double DQN)
Decouples action selection (online network) from action evaluation (target network)
to mitigate max-operator Q-value overestimation bias.
"""

from __future__ import annotations

from typing import Sequence
import torch

from agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN uses:
      a* = argmax_a Q_online(s', a)
      Y  = r + gamma * Q_target(s', a*) * (1 - done)
    """

    level = 5
    name = "Double DQN"

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.99,
        buffer_capacity: int = 25000,
        batch_size: int = 64,
        tau: float = 0.01,
        seed: int = 42,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            tau=tau,
            seed=seed,
        )
        self.level = 5
        self.name = "Double DQN"

    def compute_target_q(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions)
            return rewards + self.gamma * next_q * (1.0 - dones)
