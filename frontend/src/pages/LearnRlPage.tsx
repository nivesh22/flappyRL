import React, { useState } from 'react';
import { Sparkles, GitBranch, Layers, Compass, Award } from 'lucide-react';

export const LearnRlPage: React.FC = () => {
  const [discountGamma, setDiscountGamma] = useState<number>(0.99);
  const [surviveReward, setSurviveReward] = useState<number>(0.1);
  const [pipeReward, setPipeReward] = useState<number>(1.0);
  const [deathPenalty] = useState<number>(-10.0);

  let discountedReturn = 0;
  for (let t = 0; t < 70; t++) {
    const r = t === 69 ? surviveReward + pipeReward : surviveReward;
    discountedReturn += Math.pow(discountGamma, t) * r;
  }

  return (
    <div className="space-y-6">
      {/* Page Heading */}
      <div>
        <h1 className="text-2xl sm:text-[28px] font-bold tracking-tight text-[var(--text-primary)]">
          Learn Reinforcement Learning
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1 max-w-3xl">
          An interactive guide to the Flappy Bird Markov Decision Process (MDP), reward shaping,
          Double & Dueling Q-networks, and Proximal Policy Optimization (PPO).
        </p>
      </div>

      {/* 1. MDP Formulation & Interactive Reward Shaping Calculator */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 ui-card p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-[var(--data-blue)]" />
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              1. The five-dimensional state vector (s_t ∈ ℝ⁵)
            </h2>
          </div>
          <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
            Rather than training on raw pixel frames, the agent observes five normalized physical
            state variables at every timestep:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="ui-subcard p-3">
              <span className="font-mono font-semibold text-[var(--data-blue)]">
                s[0] = y / GROUND_Y
              </span>
              <p className="text-[var(--text-secondary)] mt-1">
                Normalized vertical position of the bird between ceiling (0.0) and ground (1.0).
              </p>
            </div>
            <div className="ui-subcard p-3">
              <span className="font-mono font-semibold text-[var(--data-blue)]">
                s[1] = v_y / MAX_VEL
              </span>
              <p className="text-[var(--text-secondary)] mt-1">
                Signed vertical velocity (−7.5 upward flap impulse, +0.45 gravity per frame).
              </p>
            </div>
            <div className="ui-subcard p-3">
              <span className="font-mono font-semibold text-[var(--data-blue)]">
                s[2] = Δx / WIDTH
              </span>
              <p className="text-[var(--text-secondary)] mt-1">
                Normalized horizontal distance to the trailing edge of the next pipe.
              </p>
            </div>
            <div className="ui-subcard p-3">
              <span className="font-mono font-semibold text-[var(--data-blue)]">
                s[3], s[4] = Δy_upper, Δy_lower
              </span>
              <p className="text-[var(--text-secondary)] mt-1">
                Signed vertical clearances to the upper and lower edges of the upcoming pipe gap.
              </p>
            </div>
          </div>
        </div>

        {/* Interactive Reward & Discount Horizon Sandbox */}
        <div className="lg:col-span-5 ui-card p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[var(--accent-primary)]" />
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              2. Interactive reward shaping calculator
            </h2>
          </div>

          <div className="ui-subcard p-4 space-y-3 text-xs">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-[var(--text-secondary)] font-medium">
                  Discount factor (γ)
                </span>
                <span className="font-mono font-semibold text-[var(--text-primary)]">
                  {discountGamma.toFixed(3)}
                </span>
              </div>
              <input
                type="range"
                aria-label="Discount factor gamma"
                min={0.8}
                max={0.999}
                step={0.005}
                value={discountGamma}
                onChange={(e) => setDiscountGamma(parseFloat(e.target.value))}
                className="w-full accent-[#F97316] cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-[var(--text-secondary)] font-medium">
                  Step survival reward
                </span>
                <span className="font-mono font-semibold text-[var(--status-success)]">
                  +{surviveReward.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                aria-label="Survive step reward"
                min={0.0}
                max={0.5}
                step={0.05}
                value={surviveReward}
                onChange={(e) => setSurviveReward(parseFloat(e.target.value))}
                className="w-full accent-[#F97316] cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <span className="text-[var(--text-secondary)] font-medium">
                  Pass pipe bonus
                </span>
                <span className="font-mono font-semibold text-[var(--data-blue)]">
                  +{pipeReward.toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                aria-label="Pass pipe bonus"
                min={0.5}
                max={5.0}
                step={0.5}
                value={pipeReward}
                onChange={(e) => setPipeReward(parseFloat(e.target.value))}
                className="w-full accent-[#F97316] cursor-pointer"
              />
            </div>

            <div className="pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between">
              <span className="text-[var(--text-secondary)] font-medium">
                70-step pipe discounted return G_t:
              </span>
              <span className="text-base font-mono font-bold text-[var(--status-success)]">
                +{discountedReturn.toFixed(2)}
              </span>
            </div>
          </div>

          <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
            Because collision incurs a <strong>{deathPenalty.toFixed(1)}</strong> penalty, a high
            discount factor (γ ≥ 0.97) is required so the agent plans 20+ frames ahead before
            flapping near an upper pipe lip.
          </p>
        </div>
      </div>

      {/* 3 Core Architectural Lessons */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="ui-card p-5 space-y-2.5">
          <div className="flex items-center gap-2 text-[var(--accent-primary)] font-semibold text-sm">
            <GitBranch className="w-4 h-4" />
            <span>Double DQN (Overestimation fix)</span>
          </div>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] leading-relaxed">
            Standard DQN uses <code className="font-mono">max_a Q(s&apos;, a; θ⁻)</code> to both
            select and evaluate actions, amplifying positive noise. Double DQN selects the winning
            action with the online network and evaluates it with the target network.
          </p>
        </div>

        <div className="ui-card p-5 space-y-2.5">
          <div className="flex items-center gap-2 text-[var(--status-success)] font-semibold text-sm">
            <Layers className="w-4 h-4" />
            <span>Dueling streams (V + A)</span>
          </div>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] leading-relaxed">
            Decomposes <code className="font-mono">Q(s,a) = V(s) + A(s,a)</code> so the network
            learns state safety <code className="font-mono">V(s)</code> on every frame without
            perturbing the narrow flap advantage threshold.
          </p>
        </div>

        <div className="ui-card p-5 space-y-2.5">
          <div className="flex items-center gap-2 text-[var(--data-blue)] font-semibold text-sm">
            <Award className="w-4 h-4" />
            <span>PPO Actor-Critic</span>
          </div>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] leading-relaxed">
            Optimizes a parameterized policy <code className="font-mono">π_θ(a|s)</code> directly
            using clipped probability ratios and Generalized Advantage Estimation (GAE), avoiding
            Q-value bootstrapping instabilities.
          </p>
        </div>
      </div>
    </div>
  );
};
