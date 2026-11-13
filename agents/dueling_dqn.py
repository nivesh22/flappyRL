"""
Level 6: Dueling Double DQN
Separates representation into a state-value stream V(s) and an action-advantage
stream A(s, a), allowing the agent to learn which states are valuable even when
action choice has minimal immediate impact (such as cruising between pipes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn import ReplayBuffer


class DuelingQNetwork(nn.Module):
    """
    Dueling Network Architecture:
      Shared trunk -> Value head V(s) [1] + Advantage head A(s, a) [action_dim]
      Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)

        trunk_layers: List[nn.Module] = []
        in_dim = state_dim
        for h_dim in self.hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(linear.bias)
            trunk_layers.append(linear)
            trunk_layers.append(nn.ReLU())
            in_dim = h_dim

        self.trunk = nn.Sequential(*trunk_layers)
        self.value_stream = nn.Linear(in_dim, 1)
        self.advantage_stream = nn.Linear(in_dim, action_dim)

        nn.init.orthogonal_(self.value_stream.weight, gain=1.0)
        nn.init.zeros_(self.value_stream.bias)
        nn.init.orthogonal_(self.advantage_stream.weight, gain=1.0)
        nn.init.zeros_(self.advantage_stream.bias)
        with torch.no_grad():
            self.advantage_stream.bias[0] = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

    def forward_streams(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values, value, advantages

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_weights_dict(self) -> Dict[str, Any]:
        trunk_data = []
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                trunk_data.append(
                    {
                        "weight": np.round(
                            module.weight.detach().cpu().numpy(), 5
                        ).tolist(),
                        "bias": np.round(
                            module.bias.detach().cpu().numpy(), 5
                        ).tolist(),
                    }
                )
        return {
            "type": "dueling",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "parameter_count": self.count_parameters(),
            "trunk": trunk_data,
            "value_head": {
                "weight": np.round(
                    self.value_stream.weight.detach().cpu().numpy(), 5
                ).tolist(),
                "bias": np.round(
                    self.value_stream.bias.detach().cpu().numpy(), 5
                ).tolist(),
            },
            "advantage_head": {
                "weight": np.round(
                    self.advantage_stream.weight.detach().cpu().numpy(), 5
                ).tolist(),
                "bias": np.round(
                    self.advantage_stream.bias.detach().cpu().numpy(), 5
                ).tolist(),
            },
        }


class DuelingDQNAgent:
    """Level 6: Dueling Double DQN Agent."""

    level = 6
    name = "Dueling Double DQN"

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
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau

        self.online_net = DuelingQNetwork(state_dim, action_dim, self.hidden_dims)
        self.target_net = DuelingQNetwork(state_dim, action_dim, self.hidden_dims)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.last_mean_q: float = 0.0

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and self.rng.random() < self.epsilon:
            return 1 if self.rng.random() < 0.16 else 0
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_net(s_t)
            return int(torch.argmax(q_values, dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.online_net(s_t).squeeze(0).cpu().numpy()

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        current_q_all = self.online_net(states)
        self.last_mean_q = float(current_q_all.max(dim=1)[0].mean().item())
        current_q = current_q_all.gather(1, actions)

        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.online_net.state_dict(),
                "hidden_dims": self.hidden_dims,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.online_net.load_state_dict(ckpt["state_dict"])
        self.target_net.load_state_dict(ckpt["state_dict"])
        self.epsilon = 0.0
