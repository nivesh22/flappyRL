import React, { useEffect, useRef, useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Trophy,
  Shuffle,
  Swords,
  Users,
} from 'lucide-react';
import { FlappyBirdEngine } from '../engine/flappyBird';
import {
  AGENT_CATALOG,
  AgentId,
  DecisionOutput,
  evaluateAgentDecision,
} from '../engine/inference';
import { GameCanvas } from '../components/GameCanvas';
import { HumanVsAiPage } from './HumanVsAiPage';
import { InfoTooltip } from '../components/InfoTooltip';

export const ModelArenaPage: React.FC = () => {
  const [arenaMode, setArenaMode] = useState<'multi_agent' | 'human_vs_ai'>('multi_agent');
  const [slotAgents, setSlotAgents] = useState<AgentId[]>([
    'random',
    'q_learning',
    'simple_dqn',
    'dueling_dqn',
  ]);
  const [seed, setSeed] = useState<number>(202);
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [speed, setSpeed] = useState<number>(0.5);
  const [showOverlay, setShowOverlay] = useState<boolean>(false);
  const [frameTick, setFrameTick] = useState<number>(0);

  const enginesRef = useRef<FlappyBirdEngine[]>([
    new FlappyBirdEngine(202),
    new FlappyBirdEngine(202),
    new FlappyBirdEngine(202),
    new FlappyBirdEngine(202),
  ]);

  const [decisions, setDecisions] = useState<DecisionOutput[]>(() =>
    slotAgents.map((id, i) =>
      evaluateAgentDecision(id, enginesRef.current[i].getState())
    )
  );

  const resetArena = (targetSeed: number = seed) => {
    enginesRef.current.forEach((eng) => eng.reset(targetSeed));
    setDecisions(
      slotAgents.map((id, i) =>
        evaluateAgentDecision(id, enginesRef.current[i].getState())
      )
    );
    setFrameTick((t) => t + 1);
  };

  useEffect(() => {
    resetArena(seed);
  }, [seed, slotAgents]);

  useEffect(() => {
    if (arenaMode !== 'multi_agent' || !isRunning) return;
    const intervalMs = speed < 1 ? Math.round(16.67 / speed) : 16.67;
    const steps = speed >= 1 ? Math.round(speed) : 1;

    const timer = window.setInterval(() => {
      const allDone = enginesRef.current.every((e) => e.done);
      if (allDone) {
        const nextSeed = ((seed * 1664525 + 1013904223) >>> 0) % 8000 + 100;
        setSeed(nextSeed);
        return;
      }

      const nextDecs: DecisionOutput[] = [...decisions];
      for (let s = 0; s < steps; s++) {
        enginesRef.current.forEach((eng, idx) => {
          if (!eng.done) {
            const dec = evaluateAgentDecision(slotAgents[idx], eng.getState());
            eng.step(dec.action);
            nextDecs[idx] = dec;
          }
        });
      }
      setDecisions(nextDecs);
      setFrameTick((t) => t + 1);
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [arenaMode, isRunning, speed, slotAgents, seed, decisions]);

  const handleAgentSlotChange = (slotIdx: number, newAgent: AgentId) => {
    const updated = [...slotAgents];
    updated[slotIdx] = newAgent;
    setSlotAgents(updated);
  };

  const leaderboard = slotAgents
    .map((id, idx) => {
      const meta = AGENT_CATALOG.find((a) => a.id === id) ?? AGENT_CATALOG[0];
      const eng = enginesRef.current[idx];
      return {
        slotIdx: idx,
        meta,
        score: eng.score,
        steps: eng.steps,
        reward: eng.cumulativeReward,
        done: eng.done,
        decision: decisions[idx],
      };
    })
    .sort((a, b) => b.score - a.score || b.steps - a.steps);

  return (
    <div className="space-y-6">
      {/* Page Heading & Mode Switcher (Multi-Agent vs Human vs AI) */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-[30px] font-bold tracking-tight text-[var(--text-primary)]">
            Model Arena
          </h1>
          <p className="text-sm sm:text-base text-[var(--text-secondary)] mt-1">
            Compare multiple reinforcement learning architectures side by side or challenge an AI
            agent directly under identical pipe seeds.
          </p>
        </div>

        {/* Mode Segmented Control */}
        <div className="ui-segmented self-start sm:self-auto" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={arenaMode === 'multi_agent'}
            onClick={() => setArenaMode('multi_agent')}
            className={`ui-segmented-item px-3.5 ${
              arenaMode === 'multi_agent' ? 'ui-segmented-item-brand' : ''
            }`}
          >
            <Users className="w-4 h-4" />
            <span>Multi-agent comparison</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={arenaMode === 'human_vs_ai'}
            onClick={() => setArenaMode('human_vs_ai')}
            className={`ui-segmented-item px-3.5 ${
              arenaMode === 'human_vs_ai' ? 'ui-segmented-item-brand' : ''
            }`}
          >
            <Swords className="w-4 h-4" />
            <span>Human vs AI</span>
          </button>
        </div>
      </div>

      {arenaMode === 'human_vs_ai' ? (
        <HumanVsAiPage />
      ) : (
        <>
          {/* Multi-Agent Playback Controls Bar */}
          <div className="ui-card p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-2.5">
              <button
                type="button"
                onClick={() => setIsRunning((r) => !r)}
                className="ui-btn ui-btn-primary min-w-[104px]"
              >
                {isRunning ? (
                  <>
                    <Pause className="w-4 h-4" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>Play</span>
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={() => resetArena(seed)}
                className="ui-btn ui-btn-secondary"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Restart race</span>
              </button>

              <div className="flex items-center gap-2 pl-2 border-l border-[var(--border-subtle)]">
                <label
                  htmlFor="arena-seed"
                  className="text-xs font-medium text-[var(--text-secondary)]"
                >
                  Seed
                </label>
                <input
                  id="arena-seed"
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Math.max(1, Number(e.target.value) || 202))}
                  className="w-20 h-9 px-2.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-strong)] font-mono text-xs text-[var(--text-primary)]"
                />
                <button
                  type="button"
                  onClick={() =>
                    setSeed(((seed * 1664525 + 1013904223) >>> 0) % 8000 + 100)
                  }
                  className="ui-btn ui-btn-secondary"
                >
                  <Shuffle className="w-3.5 h-3.5" />
                  <span>New seed</span>
                </button>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <label className="inline-flex items-center gap-2 text-xs text-[var(--text-secondary)] cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showOverlay}
                  onChange={(e) => setShowOverlay(e.target.checked)}
                  className="rounded accent-[#F97316] w-4 h-4"
                />
                <span>Debug overlays</span>
                <InfoTooltip
                  label="Debug overlays"
                  text="Show target pipe gap and sensor vectors inside each lane."
                />
              </label>

              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-[var(--text-secondary)]">
                  Speed
                </span>
                <div className="ui-segmented">
                  {[0.5, 1, 2, 4].map((spd) => (
                    <button
                      key={spd}
                      type="button"
                      onClick={() => setSpeed(spd)}
                      className={`ui-segmented-item font-mono text-xs px-2.5 ${
                        speed === spd ? 'ui-segmented-item-brand' : ''
                      }`}
                    >
                      {spd}×
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* 4 Side-by-Side Agent Lanes */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
            {slotAgents.map((agentId, idx) => {
              const meta =
                AGENT_CATALOG.find((a) => a.id === agentId) ?? AGENT_CATALOG[0];
              const eng = enginesRef.current[idx];
              const dec = decisions[idx];

              return (
                <div
                  key={idx}
                  className="ui-card p-4 flex flex-col justify-between space-y-3"
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-[var(--text-secondary)]">
                        Lane {idx + 1}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          eng.done
                            ? 'bg-[var(--status-danger-soft)] text-[var(--status-danger)]'
                            : 'bg-[var(--status-success-soft)] text-[var(--status-success)]'
                        }`}
                      >
                        {eng.done ? 'Terminated' : 'Active'}
                      </span>
                    </div>

                    <select
                      aria-label={`Select agent for lane ${idx + 1}`}
                      value={agentId}
                      onChange={(e) =>
                        handleAgentSlotChange(idx, e.target.value as AgentId)
                      }
                      className="w-full h-9 px-2.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-strong)] text-xs font-medium text-[var(--text-primary)] cursor-pointer"
                    >
                      {AGENT_CATALOG.map((a) => (
                        <option key={a.id} value={a.id}>
                          Level {a.level}: {a.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <GameCanvas
                    engine={eng}
                    decision={dec}
                    agentName={meta.shortName}
                    agentColor="#F7D51D"
                    showExplainability={showOverlay}
                    compact={true}
                    frameTick={frameTick}
                  />

                  <div className="ui-subcard p-3 grid grid-cols-3 gap-2 text-center">
                    <div>
                      <span className="block text-[11px] text-[var(--text-secondary)]">
                        Score
                      </span>
                      <span className="text-sm font-mono font-bold text-[var(--text-primary)]">
                        {eng.score}
                      </span>
                    </div>
                    <div>
                      <span className="block text-[11px] text-[var(--text-secondary)]">
                        Reward
                      </span>
                      <span className="text-sm font-mono font-bold text-[var(--text-primary)]">
                        {eng.cumulativeReward.toFixed(1)}
                      </span>
                    </div>
                    <div>
                      <span className="block text-[11px] text-[var(--text-secondary)]">
                        Action
                      </span>
                      <span
                        className={`text-xs font-mono font-semibold ${
                          dec?.action === 1
                            ? 'text-[var(--accent-primary)]'
                            : 'text-[var(--status-success)]'
                        }`}
                      >
                        {dec?.action === 1 ? 'Flap' : 'Glide'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Synchronized Leaderboard Table */}
          <div className="ui-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Trophy className="w-4 h-4 text-[var(--accent-primary)]" />
                <h2 className="text-base font-semibold text-[var(--text-primary)]">
                  Live race standings (Seed #{seed})
                </h2>
              </div>
              <span className="text-xs text-[var(--text-secondary)]">
                Identical obstacle sequence across all 4 lanes
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="border-b border-[var(--border-subtle)] text-xs font-semibold text-[var(--text-secondary)]">
                    <th className="py-2.5 px-3">Rank</th>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-3">Architecture</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">Pipes cleared</th>
                    <th className="py-2.5 px-3">Survival steps</th>
                    <th className="py-2.5 px-3">Cumulative reward</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-subtle)]">
                  {leaderboard.map((row, rankIdx) => (
                    <tr
                      key={row.slotIdx}
                      className="hover:bg-[var(--bg-subtle)]/60 transition-colors"
                    >
                      <td className="py-3 px-3 font-mono font-semibold text-[var(--accent-primary)]">
                        #{rankIdx + 1}
                      </td>
                      <td className="py-3 px-3 font-semibold text-[var(--text-primary)]">
                        {row.meta.name}
                      </td>
                      <td className="py-3 px-3 font-mono text-xs text-[var(--text-secondary)]">
                        {row.meta.architecture}
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                            row.done
                              ? 'bg-[var(--status-danger-soft)] text-[var(--status-danger)]'
                              : 'bg-[var(--status-success-soft)] text-[var(--status-success)]'
                          }`}
                        >
                          {row.done ? 'Terminated' : 'Active'}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-mono font-bold text-[var(--text-primary)]">
                        {row.score}
                      </td>
                      <td className="py-3 px-3 font-mono text-xs text-[var(--text-secondary)]">
                        {row.steps}
                      </td>
                      <td className="py-3 px-3 font-mono text-xs text-[var(--data-blue)] font-semibold">
                        {row.reward >= 0 ? `+${row.reward.toFixed(1)}` : row.reward.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
