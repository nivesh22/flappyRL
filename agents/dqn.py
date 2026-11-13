"""
Levels 3 & 4: Deep Q-Network (DQN) from Scratch
Supports configurable hidden layer architectures:
  - Tiny DQN:   [8]
  - Small DQN:  [32]
  - Medium DQN: [64, 32]
  - Deep DQN:   [128, 64, 32]
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ARCHITECTURES: Dict[str, List[int]] = {
    "tiny": [8],
    "small": [32],
    "medium": [64, 32],
    "deep": [128, 64, 32],
}


class QNetwork(nn.Module):
    """Multi-Layer Perceptron Q-Network mapping 5D state -> 2 action Q-values."""

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

        layers: List[nn.Module] = []
        in_dim = state_dim
        for h_dim in self.hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = h_dim

        out_layer = nn.Linear(in_dim, action_dim)
        nn.init.orthogonal_(out_layer.weight, gain=1.0)
        nn.init.zeros_(out_layer.bias)
        # Slight initial preference for action 0 (do nothing) so untrained bird falls naturally
        with torch.no_grad():
            out_layer.bias[0] = 0.1
        layers.append(out_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        activations: List[np.ndarray] = []
        h = x
        for module in self.net:
            h = module(h)
            if isinstance(module, nn.ReLU):
                activations.append(h.detach().cpu().numpy().squeeze(0))
        return h, activations

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_weights_dict(self) -> Dict[str, Any]:
        layers_data = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                layers_data.append(
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
            "type": "mlp",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "parameter_count": self.count_parameters(),
            "layers": layers_data,
        }


class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    def __init__(self, capacity: int = 25000, seed: int = 42):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Standard Deep Q-Network Agent (Levels 3 & 4).
    Uses Experience Replay and a periodic/soft Target Network update.
    """

    level = 3
    name = "Simple DQN"

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dims: Sequence[int] = (32,),
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

        if len(self.hidden_dims) >= 3:
            self.level = 4
            self.name = "Deep DQN"

        self.online_net = QNetwork(state_dim, action_dim, self.hidden_dims)
        self.target_net = QNetwork(state_dim, action_dim, self.hidden_dims)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.last_mean_q: float = 0.0

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and self.rng.random() < self.epsilon:
            # Biased random exploration so bird doesn't spam flap to the ceiling
            return 1 if self.rng.random() < 0.16 else 0
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_net(s_t)
            return int(torch.argmax(q_values, dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.online_net(s_t).squeeze(0).cpu().numpy()

    def compute_target_q(
        self, next_states: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            return rewards + self.gamma * max_next_q * (1.0 - dones)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        current_q_all = self.online_net(states)
        self.last_mean_q = float(current_q_all.max(dim=1)[0].mean().item())
        current_q = current_q_all.gather(1, actions)
        target_q = self.compute_target_q(next_states, rewards, dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Soft update target network
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
