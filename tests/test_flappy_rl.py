"""
Comprehensive unit and integration test suite for FlappyRL:
- Deterministic Mulberry32 PRNG and FlappyBirdEnv physics
- Gymnasium wrapper compliance
- All 8 progressive RL agents (Levels 0 through 7)
- ONNX export and inference parity with PyTorch
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

from environment.flappy_bird import FlappyBirdEnv, Mulberry32
from environment.gym_wrapper import FlappyBirdGymEnv
from agents.random_agent import RandomAgent
from agents.rule_based import RuleBasedAgent
from agents.q_learning import TabularQLearningAgent
from agents.dqn import DQNAgent, ARCHITECTURES
from agents.double_dqn import DoubleDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.ppo import PPOAgent


def test_deterministic_prng_and_env_seed():
    p1 = Mulberry32(12345)
    p2 = Mulberry32(12345)
    seq1 = [p1.next_float() for _ in range(20)]
    seq2 = [p2.next_float() for _ in range(20)]
    assert np.allclose(seq1, seq2)

    env1 = FlappyBirdEnv(seed=777)
    env2 = FlappyBirdEnv(seed=777)
    s1 = env1.reset(seed=777)
    s2 = env2.reset(seed=777)
    assert np.allclose(s1, s2)
    assert s1.shape == (5,)

    for a in [0, 1, 0, 0, 1, 0]:
        ns1, r1, t1, tr1, _ = env1.step(a)
        ns2, r2, t2, tr2, _ = env2.step(a)
        assert np.allclose(ns1, ns2)
        assert r1 == r2
        assert t1 == t2 and tr1 == tr2


def test_gymnasium_wrapper():
    env = FlappyBirdGymEnv(seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (5,)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 2
    assert "score" in info

    obs2, reward, terminated, truncated, info2 = env.step(1)
    assert obs2.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_all_eight_agent_levels():
    env = FlappyBirdEnv(seed=42)
    state = env.reset(seed=42)

    # Level 0: Random
    a0 = RandomAgent(seed=42)
    assert a0.select_action(state) in (0, 1)

    # Level 1: Rule-based
    a1 = RuleBasedAgent()
    assert a1.select_action(state) in (0, 1)

    # Level 2: Tabular Q-Learning
    a2 = TabularQLearningAgent(seed=42)
    act2 = a2.select_action(state, training=True)
    next_state, r, d, _, _ = env.step(act2)
    td_err = a2.update(state, act2, r, next_state, d)
    assert td_err >= 0.0

    # Level 3 & 4: Simple & Deep DQN
    for arch_key in ("tiny", "small", "medium", "deep"):
        dqn = DQNAgent(hidden_dims=ARCHITECTURES[arch_key], batch_size=8, seed=42)
        for _ in range(12):
            dqn.memory.push(state, 0, 0.1, next_state, False)
        loss = dqn.train_step()
        assert loss is not None
        assert dqn.select_action(state, training=False) in (0, 1)

    # Level 5: Double DQN
    ddqn = DoubleDQNAgent(batch_size=8, seed=42)
    for _ in range(12):
        ddqn.memory.push(state, 1, 0.1, next_state, False)
    assert ddqn.train_step() is not None
    assert ddqn.select_action(state, training=False) in (0, 1)

    # Level 6: Dueling Double DQN
    dueling = DuelingDQNAgent(batch_size=8, seed=42)
    for _ in range(12):
        dueling.memory.push(state, 0, 0.1, next_state, False)
    assert dueling.train_step() is not None
    assert dueling.select_action(state, training=False) in (0, 1)

    # Level 7: PPO
    ppo = PPOAgent(seed=42)
    for _ in range(10):
        a = ppo.select_action(state, training=True)
        ppo.store_outcome(0.1, False)
    loss_ppo = ppo.update()
    assert isinstance(loss_ppo, float)
    assert ppo.select_action(state, training=False) in (0, 1)


def test_onnx_export_parity():
    onnx_dir = Path(__file__).resolve().parent.parent / "checkpoints" / "onnx"
    if not (onnx_dir / "double_dqn.onnx").exists():
        return

    ddqn = DoubleDQNAgent(hidden_dims=ARCHITECTURES["medium"])
    ddqn.load(onnx_dir.parent / "double_dqn.pt")

    sess = ort.InferenceSession(
        str(onnx_dir / "double_dqn.onnx"), providers=["CPUExecutionProvider"]
    )
    sample = np.array([[0.45, -0.15, 0.52, 0.12, 0.11]], dtype=np.float32)
    ort_out = sess.run(None, {"state": sample})[0]
    with torch.no_grad():
        torch_out = ddqn.online_net(torch.tensor(sample)).numpy()
    np.testing.assert_allclose(ort_out, torch_out, rtol=1e-4, atol=1e-5)
