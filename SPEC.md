# **Project Specification: Flappy Bird — Evolution of Reinforcement Learning**

The objective is to build an interactive machine learning portfolio project demonstrating how reinforcement learning agents evolve from making random decisions to mastering Flappy Bird.

Rather than simply training a high-performing model, the project should tell the story of **how an AI learns, fails, improves, and eventually masters a game.**

## **1. Project overview**

**Suggested name:** FlappyRL — From Random to Mastery

Develop a web application where users can:

- Select different reinforcement learning models.
- Watch AI agents play Flappy Bird in real time.
- Compare simple and complex neural network architectures.
- Understand how different training approaches affect performance.
- Explore training metrics, learning curves and model behavior.
- Play the game themselves and compare their performance against the AI.

## **2. Model progression**

The project should demonstrate incremental improvements across multiple models.

| Level | Model              | Purpose                                       |
| ----- | ------------------ | --------------------------------------------- |
| 0     | Random agent       | Establish baseline performance                |
| 1     | Rule-based agent   | Establish a non-ML benchmark                  |
| 2     | Tabular Q-Learning | Introduce basic reinforcement learning        |
| 3     | Simple DQN         | Introduce neural networks                     |
| 4     | Deep DQN           | Explore increased network capacity            |
| 5     | Double DQN         | Reduce Q-value overestimation                 |
| 6     | Dueling Double DQN | Explore more advanced architectures           |
| 7     | PPO                | Compare value-based and policy-based learning |

The architectures should be trained and evaluated under comparable conditions. Increasing complexity does not necessarily guarantee better performance, and that itself is an interesting result to demonstrate.

## **3. Reinforcement learning environment**

### **State space**

Initially, use numerical game-state features rather than raw image pixels.

- Bird's vertical position
- Bird's vertical velocity
- Horizontal distance to the next pipe
- Vertical distance to the upper pipe
- Vertical distance to the lower pipe

### **Action space**

Two discrete actions:

- `0`: Do nothing
- `1`: Flap wings

### **Reward function**

| Event                    | Initial reward |
| ------------------------ | -------------- |
| Survive one timestep     | +0.1           |
| Successfully pass a pipe | +1             |
| Collision or death       | −10            |

Reward shaping can itself be an experiment.

## **4. Neural network architectures**

Illustrative architecture of the initial Deep Q-Network.

| Architecture | Hidden layers |
| ------------ | ------------- |
| Tiny DQN     | 8             |
| Small DQN    | 32            |
| Medium DQN   | 64 → 32       |
| Deep DQN     | 128 → 64 → 32 |

Compare the impact of network depth, parameter count and training time while keeping the environment consistent.

## **5. User interface**

The UI should look like an interactive AI experimentation lab rather than a traditional game website.

### **Application pages**

| Page               | Functionality                                      |
| ------------------ | -------------------------------------------------- |
| Playground         | Select an agent and watch it play                  |
| Model Arena        | Run multiple agents side by side                   |
| Training Analytics | Explore learning curves and metrics                |
| Model Explorer     | Understand architectures and hyperparameters       |
| Human vs AI        | Play against trained agents                        |
| Learn RL           | Interactive explanations of reinforcement learning |

## **6. Enhanced UI features**

The features that would make this project stand out:

- **Model Arena:** Run a random agent, simple DQN and advanced agent simultaneously using the same pipe sequence.
- **Training Time Machine:** Select a checkpoint from different training stages and watch the same agent evolve (e.g., Untrained, Episode 100, Episode 500, Episode 1,000, Trained).
- **AI Decision Visualizer:** Show state variables, Q-values, selected actions and neural network activations during gameplay.
- **Game controls:** Pause, restart, adjust simulation speed and enable slow-motion playback.
- **Explainability overlay:** Draw the bird's distance to the next pipe, velocity vector and target gap directly on the game canvas.

## **7. Training analytics**

Track the following for every model:

- Mean and maximum game score
- Cumulative episode reward
- Rolling average reward
- Training time and environment steps
- Network parameter count
- Sample efficiency
- Exploration rate
- Performance variability across random seeds

Evaluate models using unseen random seeds, identical evaluation environments and disabled exploration where appropriate. Report mean performance alongside uncertainty rather than relying solely on the highest achieved score.

## **8. Recommended technology stack**

| Component               | Technology                          |
| ----------------------- | ----------------------------------- |
| Reinforcement learning  | PyTorch                             |
| Training environment    | Gymnasium                           |
| Game development        | Python / HTML Canvas (matched)      |
| Frontend                | React + TypeScript                  |
| Styling                 | Tailwind CSS                        |
| Charts                  | Recharts                            |
| Browser model inference | ONNX Runtime Web ( + JSON weights)  |
| Testing                 | Pytest + Vitest / Playwright        |
| Deployment              | Vercel (Static / Browser Inference) |

Train the models offline and export compatible trained neural networks to ONNX (and lightweight JSON/Float32 weights for instant synchronous 60fps canvas inference alongside ONNX Runtime Web). This allows users to run inference directly in their browsers without requiring an expensive GPU-backed server.

## **9. GitHub repository structure**

```
flappyRL/
├── README.md
├── SPEC.md
├── environment/
│   ├── flappy_bird.py
│   └── gym_wrapper.py
├── agents/
│   ├── random_agent.py
│   ├── rule_based.py
│   ├── q_learning.py
│   ├── dqn.py
│   ├── double_dqn.py
│   ├── dueling_dqn.py
│   └── ppo.py
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── export_models.py
│   └── configs/
├── checkpoints/
├── experiments/
├── frontend/
├── tests/
└── docs/
```

## **10. Implementation roadmap**

| Phase | Deliverable                                 |
| ----- | ------------------------------------------- |
| 1     | Flappy Bird game and Gymnasium environment  |
| 2     | Random, rule-based and Q-Learning agents    |
| 3     | DQN implementation from scratch             |
| 4     | Advanced models and comparative experiments |
| 5     | Model checkpoints and browser inference     |
| 6     | Interactive playground and model arena      |
| 7     | Training dashboard and educational content  |
| 8     | Testing, deployment and documentation       |
