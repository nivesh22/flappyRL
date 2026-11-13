"""
Level 7: Proximal Policy Optimization (PPO)
Policy-gradient Actor-Critic method using Generalized Advantage Estimation (GAE)
and a clipped surrogate objective.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network:
      forward(x) returns policy logits [batch, 2] for direct ONNX compatibility
      with the value-based networks in the browser inference engine.
      forward_actor_critic(x) returns (logits, value).
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

        actor_layers: List[nn.Module] = []
        in_dim = state_dim
        for h_dim in self.hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(linear.bias)
            actor_layers.append(linear)
            actor_layers.append(nn.Tanh())
            in_dim = h_dim
        actor_out = nn.Linear(in_dim, action_dim)
        nn.init.orthogonal_(actor_out.weight, gain=0.1)
        nn.init.zeros_(actor_out.bias)
        with torch.no_grad():
            actor_out.bias[0] = 0.6  # initial prior favoring no-flap (~65% no-flap)
        actor_layers.append(actor_out)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers: List[nn.Module] = []
        in_dim = state_dim
        for h_dim in self.hidden_dims:
            linear = nn.Linear(in_dim, h_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(linear.bias)
            critic_layers.append(linear)
            critic_layers.append(nn.Tanh())
            in_dim = h_dim
        critic_out = nn.Linear(in_dim, 1)
        nn.init.orthogonal_(critic_out.weight, gain=1.0)
        nn.init.zeros_(critic_out.bias)
        critic_layers.append(critic_out)
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns policy logits for action selection / ONNX export."""
        return self.actor(x)

    def forward_actor_critic(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_weights_dict(self) -> Dict[str, Any]:
        actor_data = []
        for module in self.actor:
            if isinstance(module, nn.Linear):
                actor_data.append(
                    {
                        "weight": np.round(
                            module.weight.detach().cpu().numpy(), 5
                        ).tolist(),
                        "bias": np.round(
                            module.bias.detach().cpu().numpy(), 5
                        ).tolist(),
                    }
                )
        critic_data = []
        for module in self.critic:
            if isinstance(module, nn.Linear):
                critic_data.append(
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
            "type": "ppo",
            "activation": "tanh",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "parameter_count": self.count_parameters(),
            "layers": actor_data,
            "critic_layers": critic_data,
        }


class PPOAgent:
    """Level 7: PPO Agent with GAE-Lambda and clipped policy ratio."""

    level = 7
    name = "PPO (Actor-Critic)"

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 2,
        hidden_dims: Sequence[int] = (64, 32),
        lr: float = 1.5e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.015,
        value_coef: float = 0.5,
        update_epochs: int = 6,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = list(hidden_dims)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs

        self.net = ActorCriticNetwork(state_dim, action_dim, self.hidden_dims)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = self.net.forward_actor_critic(s_t)
            if not training:
                return int(torch.argmax(logits, dim=1).item())
            dist = Categorical(logits=logits)
            action = dist.sample()
            self.states.append(state)
            self.actions.append(int(action.item()))
            self.log_probs.append(float(dist.log_prob(action).item()))
            self.values.append(float(value.squeeze().item()))
            return int(action.item())

    def store_outcome(self, reward: float, done: bool) -> None:
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.net(s_t)
            probs = torch.softmax(logits, dim=-1)
            return probs.squeeze(0).cpu().numpy()

    def update(self) -> float:
        if len(self.states) == 0:
            return 0.0

        # Compute GAE returns and advantages
        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0

        for i in reversed(range(len(self.rewards))):
            mask = 0.0 if self.dones[i] else 1.0
            delta = (
                self.rewards[i]
                + self.gamma * next_value * mask
                - self.values[i]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
            next_value = self.values[i]

        states_t = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.update_epochs):
            logits, values = self.net.forward_actor_critic(states_t)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratios * adv_t
            surr2 = (
                torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.smooth_l1_loss(values.squeeze(-1), returns_t)

            loss = (
                actor_loss
                + self.value_coef * critic_loss
                - self.entropy_coef * entropy
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss += float(loss.item())

        self.clear_buffer()
        return total_loss / self.update_epochs

    def clear_buffer(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "hidden_dims": self.hidden_dims,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(ckpt["state_dict"])
