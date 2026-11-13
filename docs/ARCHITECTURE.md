# FlappyRL — Technical Architecture & Experimental Design

## 1. Environment & Physics Parity (`environment/` ↔ `frontend/src/engine/`)

To ensure zero discrepancy between offline PyTorch training and live browser evaluation, both the Python environment (`environment/flappy_bird.py`) and the TypeScript HTML5 Canvas engine (`frontend/src/engine/flappyBird.ts`) share:

- **Identical Physics Constants**:
  - Canvas: `400 × 600` (`GROUND_Y = 520`)
  - Bird: `BIRD_X = 80`, `BIRD_RADIUS = 14`, `GRAVITY = 0.45`, `FLAP_STRENGTH = -7.5`, `MAX_VEL_Y = 10.0`
  - Pipes: `PIPE_WIDTH = 56`, `PIPE_GAP = 120`, `PIPE_SPEED = 3.0`, `PIPE_SPAWN_SPACING = 210`
- **Bit-Identical Portable PRNG (`Mulberry32`)**:
  - Both Python and TypeScript implement the exact 32-bit unsigned integer `Mulberry32` algorithm so any seed `S` produces the exact same sequence of pipe gap heights (`gap_top_y`).
- **5D Normalized State Space (`s_t ∈ ℝ⁵`)**:
  1. `bird_y / GROUND_Y`
  2. `bird_vel / MAX_VEL_Y`
  3. `(next_pipe.x + PIPE_WIDTH - BIRD_X) / WIDTH`
  4. `(bird_y - next_pipe.gap_top_y) / GROUND_Y`
  5. `(next_pipe.gap_bottom_y - bird_y) / GROUND_Y`

## 2. Progressive Model Hierarchy (`agents/`)

| Level | Agent | File | Architecture | Key Mechanism |
| --- | --- | --- | --- | --- |
| 0 | Random Agent | `agents/random_agent.py` | `Bernoulli(p=0.12)` | Stochastic baseline |
| 1 | Rule-Based Agent | `agents/rule_based.py` | 2-rule threshold | Hand-crafted heuristic benchmark |
| 2 | Tabular Q-Learning | `agents/q_learning.py` | `14 × 8 × 6 × 2` Q-Table | State discretization + Bellman TD update |
| 3 | Simple DQN | `agents/dqn.py` | `5 → 32 → 2` (258 params) | Experience Replay + Target Network |
| 4 | Deep DQN | `agents/dqn.py` | `5 → 128 → 64 → 32 → 2` (11,170 params) | Deep MLP capacity ablation |
| 5 | Double DQN | `agents/double_dqn.py` | `5 → 64 → 32 → 2` (2,530 params) | Decoupled action selection & evaluation |
| 6 | Dueling Double DQN | `agents/dueling_dqn.py` | `5 → 64 → 32 → [V(s) + A(s,a)]` (2,563 params) | Separate State-Value & Advantage streams |
| 7 | PPO (Actor-Critic) | `agents/ppo.py` | Actor `5→64→32→2`, Critic `5→64→32→1` | Clipped surrogate objective + GAE |

## 3. Browser Inference Architecture

All trained PyTorch networks are exported via `training/export_models.py` to:
1. Standalone `.onnx` models in `checkpoints/onnx/` and `frontend/public/models/` for `onnxruntime-web`.
2. Layer weight matrices and biases in `frontend/src/data/modelsBundle.json` for synchronous 60 FPS canvas execution and real-time hidden-layer neuron activation visualization (`NeuralNetVisualizer.tsx`).
