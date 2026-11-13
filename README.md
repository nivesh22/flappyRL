# FlappyRL — From Random to Mastery

> An interactive reinforcement learning laboratory that demonstrates how AI agents learn to master Flappy Bird, featuring progressive neural network architectures, comparative experiments, real-time model inference, and interactive training visualizations.

---

## Overview

**FlappyRL** tells the story of **how an AI learns, fails, improves, and eventually masters a game** across 8 progressive reinforcement learning levels:

| Level | Model | Architecture | Parameters | Purpose |
| ----- | ----- | ------------ | ---------- | ------- |
| **0** | **Random Agent** | `Bernoulli(p=0.12)` | 1 | Establish stochastic baseline performance |
| **1** | **Rule-Based Agent** | `Threshold Heuristic` | 2 | Establish a non-ML domain heuristic benchmark |
| **2** | **Tabular Q-Learning** | `Q-Table (14×8×6×2)` | 1,344 | Introduce Bellman TD updates & state discretization |
| **3** | **Simple DQN** | `MLP 5 → 32 → 2` | 258 | Introduce neural function approximation & replay buffers |
| **4** | **Deep DQN** | `MLP 5 → 128 → 64 → 32 → 2` | 11,170 | Explore increased network depth & capacity tradeoffs |
| **5** | **Double DQN** | `MLP 5 → 64 → 32 → 2` | 2,530 | Reduce max-operator Q-value overestimation bias |
| **6** | **Dueling Double DQN** | `5 → 64 → 32 → [V(s) + A(s,a)]` | 2,563 | Separate state value $V(s)$ from action advantage $A(s,a)$ |
| **7** | **PPO (Actor-Critic)** | `Actor 5→64→32→2 \| Critic 5→64→32→1` | 5,027 | Compare value-based and policy-gradient learning |

---

## Key Features

1. **Interactive RL Playground (`Playground`)**:
   - Select any agent from Level 0 to Level 7 and watch it play in real time.
   - **Training Time Machine**: Load saved evolutionary checkpoints (`Untrained Ep 0`, `Episode 100`, `Episode 350`, `Episode 700`, `Trained Mastery Ep 1,200`) and watch the same agent evolve from crashing immediately to threading pipes effortlessly.
   - **Live AI Decision Visualizer**: Real-time Q-values, action probabilities, 5D state vector bars, and an interactive neural network activation graph showing every hidden layer's firing pattern at 60 FPS.
   - **Explainability Overlay**: Renders horizontal/vertical pipe clearance lines, target gap center crosshair, optimal flap trigger threshold, and velocity vectors directly on the game canvas.
2. **Synchronized Model Arena (`Model Arena`)**:
   - Run 4 agents simultaneously side-by-side on the exact same deterministic pipe sequence (powered by a bit-identical `Mulberry32` PRNG shared between Python and TypeScript).
3. **Training Analytics (`Training Analytics`)**:
   - Explore multi-seed learning curves, empirical Q-value overestimation comparisons (Standard DQN vs Double DQN), and Section 4 network capacity ablations (`Tiny 8`, `Small 32`, `Medium 64→32`, `Deep 128→64→32`).
4. **Model Explorer (`Model Explorer`)**:
   - Inspect mathematical objectives, hyperparameters, and test counterfactual game states with interactive sliders across all 8 models simultaneously.
5. **Human vs AI (`Human vs AI`)**:
   - Play Flappy Bird live (`Spacebar` / `Up Arrow` / Click) head-to-head against any trained RL agent on the same pipe stream.
6. **Learn RL (`Learn RL`)**:
   - Interactive visual explanations of the Flappy Bird MDP, reward shaping, Bellman optimality, Double/Dueling DQN, and PPO.

---

## Repository Structure

```
flappyRL/
├── README.md
├── SPEC.md
├── pytest.ini
├── environment/
│   ├── flappy_bird.py       # Deterministic Flappy Bird physics + Mulberry32 PRNG
│   └── gym_wrapper.py       # Gymnasium-compatible environment wrapper
├── agents/
│   ├── random_agent.py      # Level 0: Random baseline
│   ├── rule_based.py        # Level 1: Rule-based heuristic
│   ├── q_learning.py        # Level 2: Tabular Q-Learning
│   ├── dqn.py               # Levels 3 & 4: Configurable MLP DQN (Tiny/Small/Medium/Deep)
│   ├── double_dqn.py        # Level 5: Double DQN
│   ├── dueling_dqn.py       # Level 6: Dueling Double DQN (V(s) + A(s,a))
│   └── ppo.py               # Level 7: Actor-Critic PPO
├── training/
│   ├── train.py             # Training pipeline, capacity ablation & Time Machine checkpoints
│   ├── evaluate.py          # Multi-seed unseen evaluation benchmark
│   ├── export_models.py     # PyTorch -> ONNX (.onnx) + JSON browser bundle exporter
│   └── configs/
│       └── default.json
├── checkpoints/             # Saved PyTorch (.pt), Q-table (.json), and ONNX (.onnx) models
├── experiments/
│   └── training_metrics.json
├── frontend/                # React + TypeScript + Tailwind CSS + Recharts + ONNX Runtime Web
├── tests/
│   └── test_flappy_rl.py    # Pytest suite verifying physics, agents, and ONNX parity
└── docs/
    └── ARCHITECTURE.md
```

---

## Quick Start

### 1. Python RL Environment, Training & Tests

```bash
# Install Python dependencies
pip install torch numpy gymnasium onnx onnxruntime pytest

# Train all 8 agent levels + Time Machine checkpoints and export to ONNX
python -m training.train
python -m training.export_models

# Run test suite
pytest -v
```

### 2. Interactive Web Laboratory (Frontend)

```bash
cd frontend
npm install
npm run dev
```
