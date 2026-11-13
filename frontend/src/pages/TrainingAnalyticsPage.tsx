import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { TrendingUp, Layers, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { AGENT_CATALOG, EXPERIMENTS_DATA } from '../engine/inference';
import { InfoTooltip } from '../components/InfoTooltip';

const INFERENCE_LATENCY_MS: Record<string, string> = {
  random: '0.01 ms',
  rule_based: '0.01 ms',
  q_learning: '0.02 ms',
  simple_dqn: '0.08 ms',
  deep_dqn: '0.24 ms',
  double_dqn: '0.12 ms',
  dueling_dqn: '0.14 ms',
  ppo: '0.16 ms',
};

export const TrainingAnalyticsPage: React.FC = () => {
  const [visibleModels, setVisibleModels] = useState<Record<string, boolean>>({
    random: true,
    rule_based: false,
    q_learning: true,
    simple_dqn: true,
    deep_dqn: true,
    double_dqn: true,
    dueling_dqn: true,
    ppo: true,
  });

  const curves = EXPERIMENTS_DATA?.learning_curves ?? [];
  const modelsDict = EXPERIMENTS_DATA?.models ?? {};
  const capacityDict = EXPERIMENTS_DATA?.capacity_ablation ?? {};

  const capacityData = Object.entries(capacityDict).map(([key, item]: [string, any]) => ({
    id: key,
    name: item.name,
    layers: item.hidden_layers,
    parameters: item.parameters,
    meanScore: item.evaluation?.mean_score ?? 0,
    maxScore: item.evaluation?.max_score ?? 0,
    trainTime: item.training_time_sec,
  }));

  const toggleModelLine = (id: string) => {
    setVisibleModels((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="space-y-6">
      {/* Page Heading */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-[28px] font-bold tracking-tight text-[var(--text-primary)]">
            Training & Evaluation Analytics
          </h1>
          <p className="text-sm text-[var(--text-secondary)] mt-1 max-w-3xl">
            Comparing training sample efficiency, Q-value overestimation bias, network capacity
            ablations, and held-out evaluation performance across 10 unseen deterministic seeds (ε =
            0).
          </p>
        </div>
      </div>

      {/* 1. Learning Curves Over Training */}
      <div className="ui-card p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-[var(--accent-primary)]" />
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              Rolling evaluation score across training (0 – 1,200 episodes)
            </h2>
            <InfoTooltip
              label="Training vs evaluation curves"
              text="Shows rolling average pipes cleared as training progresses from episode 0 to 1,200 under identical environment physics."
            />
          </div>

          <div className="flex flex-wrap gap-1.5">
            {AGENT_CATALOG.map((agent) => (
              <button
                key={agent.id}
                type="button"
                onClick={() => toggleModelLine(agent.id)}
                className={`ui-btn h-8 px-2.5 text-xs ${
                  visibleModels[agent.id]
                    ? 'ui-btn-secondary font-semibold'
                    : 'ui-btn-ghost opacity-60'
                }`}
              >
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: agent.color }}
                />
                <span>{agent.shortName}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="h-[320px] w-full ui-subcard p-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={curves} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis
                dataKey="episode"
                stroke="var(--text-secondary)"
                tick={{ fontSize: 12 }}
              />
              <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-card)',
                  borderColor: 'var(--border-strong)',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: 'var(--text-primary)',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              {AGENT_CATALOG.map(
                (agent) =>
                  visibleModels[agent.id] && (
                    <Line
                      key={agent.id}
                      type="monotone"
                      dataKey={agent.id}
                      name={agent.name}
                      stroke={agent.color}
                      strokeWidth={2.2}
                      dot={false}
                    />
                  )
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2. Q-Value Overestimation + Network Capacity Ablation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="ui-card p-5 space-y-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-[var(--accent-primary)]" />
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              Q-value overestimation: Standard DQN vs Double DQN
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)]">
            Standard DQN inflates predicted Q-values above the true discounted return V*(s), whereas
            Double DQN decouples action selection and evaluation to track true return.
          </p>

          <div className="h-[250px] w-full ui-subcard p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curves}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                <XAxis
                  dataKey="episode"
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 12 }}
                />
                <YAxis
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 12 }}
                  domain={[0, 10]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--bg-card)',
                    borderColor: 'var(--border-strong)',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line
                  type="monotone"
                  dataKey="dqn_mean_q"
                  name="Standard DQN Mean Q"
                  stroke="#F97316"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="double_dqn_mean_q"
                  name="Double DQN Mean Q"
                  stroke="#16A34A"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="true_return"
                  name="True Discounted Return V*(s)"
                  stroke="#0284C7"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="ui-card p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[var(--data-blue)]" />
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              Network capacity ablation (Tiny to Deep DQN)
            </h2>
          </div>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)]">
            Comparing Tiny (8), Small (32), Medium (64→32), and Deep (128→64→32) hidden-layer
            capacities under identical training budgets.
          </p>

          <div className="h-[250px] w-full ui-subcard p-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={capacityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                <XAxis
                  dataKey="name"
                  stroke="var(--text-secondary)"
                  tick={{ fontSize: 12 }}
                />
                <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--bg-card)',
                    borderColor: 'var(--border-strong)',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar
                  dataKey="meanScore"
                  name="Mean evaluation score"
                  fill="#0284C7"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="maxScore"
                  name="Max evaluation score"
                  fill="#F97316"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 3. Comprehensive Held-Out Evaluation Benchmark Table */}
      <div className="ui-card p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-[var(--status-success)]" />
            <h2 className="text-base font-semibold text-[var(--text-primary)]">
              Held-out evaluation benchmark (10 unseen random seeds, ε = 0)
            </h2>
          </div>
          <span className="text-xs text-[var(--text-secondary)]">
            Distinguishes held-out evaluation performance from noisy training rollouts
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="border-b border-[var(--border-subtle)] text-xs font-semibold text-[var(--text-secondary)]">
                <th className="py-2.5 px-3">Level</th>
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Architecture</th>
                <th className="py-2.5 px-3">Parameters</th>
                <th className="py-2.5 px-3">Eval mean ± std</th>
                <th className="py-2.5 px-3">Max pipes</th>
                <th className="py-2.5 px-3">≥10 pipes rate</th>
                <th className="py-2.5 px-3">Mean reward</th>
                <th className="py-2.5 px-3">Train duration</th>
                <th className="py-2.5 px-3">Inference latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)]">
              {AGENT_CATALOG.map((meta) => {
                const m = modelsDict[meta.id];
                const ev = m?.evaluation ?? {
                  mean_score: 0,
                  std_score: 0,
                  max_score: 0,
                  mean_reward: 0,
                  mean_steps: 0,
                  per_seed_scores: [],
                };
                const scores: number[] = ev.per_seed_scores ?? [];
                const pass10Rate =
                  scores.length > 0
                    ? Math.round(
                        (scores.filter((s) => s >= 10).length / scores.length) * 100
                      )
                    : 0;

                return (
                  <tr
                    key={meta.id}
                    className="hover:bg-[var(--bg-subtle)]/60 transition-colors"
                  >
                    <td className="py-3 px-3 font-mono font-semibold text-[var(--accent-primary)]">
                      L{meta.level}
                    </td>
                    <td className="py-3 px-3 font-semibold text-[var(--text-primary)]">
                      {meta.name}
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-[var(--text-secondary)]">
                      {meta.architecture}
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-[var(--text-primary)]">
                      {meta.parameters.toLocaleString()}
                    </td>
                    <td className="py-3 px-3 font-mono font-semibold text-[var(--status-success)]">
                      {Number(ev.mean_score).toFixed(1)} ± {Number(ev.std_score).toFixed(1)}
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-[var(--text-primary)]">
                      {ev.max_score}
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-[var(--data-blue)] font-semibold">
                      {pass10Rate}%
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-[var(--text-secondary)]">
                      {Number(ev.mean_reward).toFixed(1)}
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-[var(--text-secondary)]">
                      {m?.training_time_sec ? `${m.training_time_sec}s` : '0.0s'}
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-[var(--text-secondary)]">
                      {INFERENCE_LATENCY_MS[meta.id] ?? '0.12 ms'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
