import React, { useState } from 'react';
import { Sliders, CheckCircle2, ArrowUp, ArrowRight } from 'lucide-react';
import {
  AGENT_CATALOG,
  AgentId,
  evaluateAgentDecision,
} from '../engine/inference';

const HYPERPARAMS_BY_MODEL: Record<
  AgentId,
  {
    objective: string;
    exploration: string;
    learningRate: string;
    discount: string;
    keyInsight: string;
  }
> = {
  random: {
    objective: 'π(a=1 | s) = p = 0.12',
    exploration: '100% Stochastic Bernoulli',
    learningRate: 'None (untrained)',
    discount: 'γ = 0.0',
    keyInsight:
      'Establishes the zero-intelligence floor. Even with gravity-balanced flap frequency (12%), random actions fail at the first pipe.',
  },
  rule_based: {
    objective: 'a = 1 if (Δy_lower < 0.037 and v_y ≥ 0.04) else 0',
    exploration: 'Deterministic Heuristic',
    learningRate: 'None (hand-tuned)',
    discount: 'Myopic (1-step lookahead)',
    keyInsight:
      'Shows that domain rules can clear pipes but remain brittle when extreme vertical pipe transitions require multi-step trajectory planning.',
  },
  q_learning: {
    objective:
      "Q(b(s), a) ← Q(b(s), a) + α [r + γ max_a' Q(b(s'), a') − Q(b(s), a)]",
    exploration: 'ε-greedy (1.0 → 0.01)',
    learningRate: 'α = 0.15',
    discount: 'γ = 0.98',
    keyInsight:
      'Demonstrates the curse of dimensionality: coarse state binning causes state aliasing near pipe lips, while finer bins explode table size.',
  },
  tiny_dqn: {
    objective: "L(θ) = E[(r + γ max_a' Q(s', a'; θ⁻) − Q(s, a; θ))²]",
    exploration: 'ε-greedy + Replay Buffer',
    learningRate: '3e-3 (Adam)',
    discount: 'γ = 0.99',
    keyInsight:
      'With only 8 hidden neurons (66 parameters), Tiny DQN learns basic survival and pipe alignment remarkably fast.',
  },
  simple_dqn: {
    objective: "L(θ) = SmoothL1(r + γ max_a' Q(s', a'; θ⁻) − Q(s, a; θ))",
    exploration: 'ε-greedy (25k Replay Buffer)',
    learningRate: '2.5e-3 (Adam)',
    discount: 'γ = 0.99',
    keyInsight:
      'Replaces discrete bins with a continuous 32-unit ReLU hidden layer (258 parameters), generalizing smoothly across unseen pipe heights.',
  },
  deep_dqn: {
    objective: "L(θ) = SmoothL1(r + γ max_a' Q(s', a'; θ⁻) − Q(s, a; θ))",
    exploration: 'ε-greedy (25k Replay Buffer)',
    learningRate: '2.0e-3 (Adam)',
    discount: 'γ = 0.99',
    keyInsight:
      'At 11,170 parameters (128→64→32), increased depth requires more gradient updates to stabilize and is more prone to Q-value overestimation.',
  },
  double_dqn: {
    objective: "Y_t = r + γ Q(s', argmax_a Q(s', a; θ); θ⁻)",
    exploration: 'ε-greedy + Soft Target (τ=0.01)',
    learningRate: '2.5e-3 (Adam)',
    discount: 'γ = 0.99',
    keyInsight:
      'Decoupling action selection (θ) from evaluation (θ⁻) eliminates positive maximization bias and yields stable pipe navigation.',
  },
  dueling_dqn: {
    objective: "Q(s, a) = V(s) + (A(s, a) − 1/|A| Σ A(s, a'))",
    exploration: 'ε-greedy + Dueling Streams',
    learningRate: '2.5e-3 (Adam)',
    discount: 'γ = 0.99',
    keyInsight:
      'Separating V(s) from A(s,a) lets every timestep update state value V(s) even when gliding between pipes has near-zero action advantage.',
  },
  ppo: {
    objective:
      'L_CLIP(θ) = E[min(r_t(θ) Â_t, clip(r_t(θ), 1−ε, 1+ε) Â_t)] + c_1 L_VF − c_2 H(π)',
    exploration: 'On-Policy Categorical Entropy',
    learningRate: '2.5e-3 (Adam, GAE λ=0.95)',
    discount: 'γ = 0.99',
    keyInsight:
      'Optimizes policy probabilities π(a|s) directly with clipped trust-region updates, producing smooth stochastic action distributions.',
  },
};

export const ModelExplorerPage: React.FC = () => {
  const [selectedId, setSelectedId] = useState<AgentId>('double_dqn');
  const [probeState, setProbeState] = useState<number[]>([
    0.48,
    0.25,
    0.35,
    0.19,
    0.032,
  ]);

  const activeMeta =
    AGENT_CATALOG.find((a) => a.id === selectedId) ?? AGENT_CATALOG[5];
  const details = HYPERPARAMS_BY_MODEL[selectedId];

  const updateProbeDim = (idx: number, val: number) => {
    const next = [...probeState];
    next[idx] = val;
    setProbeState(next);
  };

  return (
    <div className="space-y-6">
      {/* Page Heading */}
      <div>
        <h1 className="text-2xl sm:text-[28px] font-bold tracking-tight text-[var(--text-primary)]">
          Model Architecture & Decision Explorer
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1 max-w-3xl">
          Inspect training objectives, network architectures, parameter counts, and counterfactual
          action decisions across all eight models (Levels 0–7).
        </p>
      </div>

      {/* Model Selector Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
        {AGENT_CATALOG.map((agent) => {
          const active = agent.id === selectedId;
          return (
            <button
              key={agent.id}
              type="button"
              onClick={() => setSelectedId(agent.id)}
              className={`p-3 rounded-xl border text-left transition-all cursor-pointer ${
                active
                  ? 'bg-[var(--accent-primary-soft)] border-[var(--accent-primary)]'
                  : 'bg-[var(--bg-card)] border-[var(--border-subtle)] hover:bg-[var(--bg-subtle)]'
              }`}
            >
              <span
                className={`text-[11px] font-mono font-semibold block ${
                  active ? 'text-[var(--accent-primary)]' : 'text-[var(--text-secondary)]'
                }`}
              >
                Level {agent.level}
              </span>
              <span className="text-xs font-semibold text-[var(--text-primary)] block mt-0.5 truncate">
                {agent.shortName}
              </span>
              <span className="text-[11px] font-mono text-[var(--text-secondary)] block mt-1">
                {agent.parameters.toLocaleString()} params
              </span>
            </button>
          );
        })}
      </div>

      {/* Selected Model Specification + Counterfactual State Probe */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 ui-card p-5 space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-xs font-semibold text-[var(--accent-primary)]">
                {activeMeta.badge}
              </span>
              <h2 className="text-xl font-bold text-[var(--text-primary)] mt-0.5">
                {activeMeta.name}
              </h2>
            </div>
            <div className="ui-subcard px-3 py-1.5 text-right font-mono">
              <span className="text-[10px] text-[var(--text-secondary)] block">
                Parameters
              </span>
              <span className="text-base font-bold text-[var(--text-primary)]">
                {activeMeta.parameters.toLocaleString()}
              </span>
            </div>
          </div>

          <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
            {activeMeta.description}
          </p>

          <div className="ui-subcard p-3.5 space-y-1 font-mono text-xs">
            <div className="text-[var(--text-secondary)] text-[11px] font-sans font-semibold">
              Optimization objective
            </div>
            <div className="text-[var(--data-blue)] break-words font-semibold">
              {details.objective}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="ui-subcard p-3">
              <span className="text-[var(--text-secondary)] block">Topology</span>
              <span className="font-mono font-semibold text-[var(--text-primary)] mt-1 block">
                {activeMeta.architecture}
              </span>
            </div>
            <div className="ui-subcard p-3">
              <span className="text-[var(--text-secondary)] block">Learning rate</span>
              <span className="font-mono font-semibold text-[var(--text-primary)] mt-1 block">
                {details.learningRate}
              </span>
            </div>
            <div className="ui-subcard p-3">
              <span className="text-[var(--text-secondary)] block">Exploration</span>
              <span className="font-mono font-semibold text-[var(--text-primary)] mt-1 block">
                {details.exploration}
              </span>
            </div>
          </div>

          <div className="p-3.5 rounded-lg bg-[var(--accent-primary-soft)] border border-[var(--accent-primary)]/30 flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-[var(--accent-primary)] shrink-0 mt-0.5" />
            <div className="text-xs text-[var(--text-primary)] leading-relaxed space-y-1">
              <div>
                <strong>Key takeaway: </strong>
                {details.keyInsight}
              </div>
              <div className="text-[var(--text-secondary)]">
                <strong>Limitation: </strong>
                {activeMeta.limitations}
              </div>
            </div>
          </div>
        </div>

        {/* Counterfactual State Probe */}
        <div className="lg:col-span-6 ui-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[var(--data-blue)]" />
              <h2 className="text-base font-semibold text-[var(--text-primary)]">
                Interactive counterfactual state probe
              </h2>
            </div>
            <span className="text-xs text-[var(--text-secondary)]">
              Compare all 8 models on the same input state
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 ui-subcard p-3.5">
            {[
              { idx: 4, label: 'Distance to lower edge (Δy_l)', min: -0.05, max: 0.2 },
              { idx: 1, label: 'Vertical velocity (v_y)', min: -0.8, max: 0.9 },
              { idx: 2, label: 'Distance to next pipe (Δx)', min: 0.0, max: 0.8 },
              { idx: 3, label: 'Distance to upper edge (Δy_u)', min: 0.0, max: 0.25 },
            ].map((slider) => (
              <div key={slider.idx}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-[var(--text-secondary)] font-medium">
                    {slider.label}
                  </span>
                  <span className="font-mono font-semibold text-[var(--text-primary)]">
                    {probeState[slider.idx] >= 0
                      ? `+${probeState[slider.idx].toFixed(3)}`
                      : probeState[slider.idx].toFixed(3)}
                  </span>
                </div>
                <input
                  type="range"
                  aria-label={slider.label}
                  min={slider.min}
                  max={slider.max}
                  step={0.005}
                  value={probeState[slider.idx]}
                  onChange={(e) =>
                    updateProbeDim(slider.idx, parseFloat(e.target.value))
                  }
                  className="w-full accent-[#F97316] cursor-pointer"
                />
              </div>
            ))}
          </div>

          <div className="space-y-1.5">
            {AGENT_CATALOG.map((agent) => {
              const out = evaluateAgentDecision(agent.id, probeState);
              const isProb =
                out.policyType === 'policy_gradient' ||
                out.policyType === 'stochastic';
              return (
                <div
                  key={agent.id}
                  className="flex items-center justify-between ui-subcard px-3 py-2 text-xs"
                >
                  <span className="font-semibold text-[var(--text-primary)] w-40 truncate">
                    L{agent.level}: {agent.name}
                  </span>
                  <span className="font-mono text-[var(--text-secondary)]">
                    {isProb
                      ? `π(glide)=${(out.actionProbs[0] * 100).toFixed(0)}% | π(flap)=${(
                          out.actionProbs[1] * 100
                        ).toFixed(0)}%`
                      : `Q(glide)=${out.qValues[0].toFixed(2)} | Q(flap)=${out.qValues[1].toFixed(
                          2
                        )}`}
                  </span>
                  <span
                    className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full font-semibold ${
                      out.action === 1
                        ? 'bg-[var(--accent-primary-soft)] text-[var(--accent-primary)]'
                        : 'bg-[var(--status-success-soft)] text-[var(--status-success)]'
                    }`}
                  >
                    {out.action === 1 ? (
                      <>
                        <ArrowUp className="w-3 h-3" />
                        <span>Flap</span>
                      </>
                    ) : (
                      <>
                        <ArrowRight className="w-3 h-3" />
                        <span>Glide</span>
                      </>
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
