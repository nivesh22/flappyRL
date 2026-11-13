import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Play, RotateCcw, Trophy, Gamepad2, Cpu } from 'lucide-react';
import { FlappyBirdEngine } from '../engine/flappyBird';
import {
  AGENT_CATALOG,
  AgentId,
  evaluateAgentDecision,
} from '../engine/inference';
import { GameCanvas } from '../components/GameCanvas';

export const HumanVsAiPage: React.FC = () => {
  const [aiOpponent, setAiOpponent] = useState<AgentId>('double_dqn');
  const [seed, setSeed] = useState<number>(303);
  const [matchRunning, setMatchRunning] = useState<boolean>(false);
  const [humanBest, setHumanBest] = useState<number>(0);
  const [aiBest, setAiBest] = useState<number>(0);
  const [frameTick, setFrameTick] = useState<number>(0);

  const humanEngineRef = useRef<FlappyBirdEngine>(new FlappyBirdEngine(303));
  const aiEngineRef = useRef<FlappyBirdEngine>(new FlappyBirdEngine(303));
  const pendingHumanFlapRef = useRef<boolean>(false);

  const aiMeta =
    AGENT_CATALOG.find((a) => a.id === aiOpponent) ?? AGENT_CATALOG[5];

  const triggerHumanFlap = useCallback(() => {
    if (!matchRunning) {
      setMatchRunning(true);
    }
    pendingHumanFlapRef.current = true;
  }, [matchRunning]);

  const resetMatch = (targetSeed: number = seed) => {
    setMatchRunning(false);
    pendingHumanFlapRef.current = false;
    humanEngineRef.current.reset(targetSeed);
    aiEngineRef.current.reset(targetSeed);
    setFrameTick((t) => t + 1);
  };

  useEffect(() => {
    resetMatch(seed);
  }, [aiOpponent, seed]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT') return;
      if (e.code === 'Space' || e.code === 'ArrowUp' || e.code === 'KeyW') {
        e.preventDefault();
        triggerHumanFlap();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [triggerHumanFlap]);

  useEffect(() => {
    if (!matchRunning) return;

    const timer = window.setInterval(() => {
      const hEng = humanEngineRef.current;
      const aEng = aiEngineRef.current;

      if (hEng.done && aEng.done) {
        setMatchRunning(false);
        return;
      }

      if (!hEng.done) {
        const hAction: 0 | 1 = pendingHumanFlapRef.current ? 1 : 0;
        pendingHumanFlapRef.current = false;
        hEng.step(hAction);
        if (hEng.score > humanBest) setHumanBest(hEng.score);
      }

      if (!aEng.done) {
        const aiDec = evaluateAgentDecision(aiOpponent, aEng.getState());
        aEng.step(aiDec.action);
        if (aEng.score > aiBest) setAiBest(aEng.score);
      }

      setFrameTick((t) => t + 1);
    }, 20);

    return () => window.clearInterval(timer);
  }, [matchRunning, aiOpponent, humanBest, aiBest]);

  const hScore = humanEngineRef.current.score;
  const aScore = aiEngineRef.current.score;

  return (
    <div className="space-y-6">
      {/* Match Controls Bar */}
      <div className="ui-card p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-[var(--text-primary)]">
            Human vs AI head-to-head challenge
          </h2>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-0.5">
            Press <kbd className="px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-strong)] font-mono text-xs">Space</kbd>,{' '}
            <kbd className="px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-strong)] font-mono text-xs">↑</kbd>, or click your game viewport to flap. Both players face Seed #{seed}.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <select
            aria-label="Select AI opponent"
            value={aiOpponent}
            onChange={(e) => setAiOpponent(e.target.value as AgentId)}
            className="h-9 px-3 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-strong)] text-xs font-medium text-[var(--text-primary)] cursor-pointer"
          >
            {AGENT_CATALOG.map((a) => (
              <option key={a.id} value={a.id}>
                Opponent: Level {a.level} — {a.name}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={triggerHumanFlap}
            className="ui-btn ui-btn-primary"
          >
            <Play className="w-4 h-4" />
            <span>{matchRunning ? 'Flap (Space)' : 'Start match'}</span>
          </button>

          <button
            type="button"
            onClick={() => resetMatch(seed)}
            className="ui-btn ui-btn-secondary"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Match KPI Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="ui-card p-4 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-[var(--text-secondary)]">
              Human player (You)
            </span>
            <div className="text-3xl font-bold font-mono text-[var(--text-primary)] mt-1">
              {hScore}
            </div>
          </div>
          <div className="text-right font-mono text-xs text-[var(--text-secondary)]">
            <div>Best: {humanBest}</div>
            <div>Steps: {humanEngineRef.current.steps}</div>
          </div>
        </div>

        <div className="ui-card p-4 flex flex-col items-center justify-center text-center">
          <Trophy className="w-5 h-5 text-[var(--accent-primary)] mb-1" />
          <span className="text-xs font-medium text-[var(--text-secondary)]">
            Live match status
          </span>
          <span className="text-sm font-semibold text-[var(--text-primary)] mt-0.5">
            {hScore > aScore
              ? `You lead by +${hScore - aScore}`
              : aScore > hScore
              ? `${aiMeta.name} leads by +${aScore - hScore}`
              : 'Score tied'}
          </span>
        </div>

        <div className="ui-card p-4 flex items-center justify-between">
          <div>
            <span className="text-xs font-medium text-[var(--text-secondary)]">
              AI opponent ({aiMeta.name})
            </span>
            <div className="text-3xl font-bold font-mono text-[var(--accent-primary)] mt-1">
              {aScore}
            </div>
          </div>
          <div className="text-right font-mono text-xs text-[var(--text-secondary)]">
            <div>Best: {aiBest}</div>
            <div>Steps: {aiEngineRef.current.steps}</div>
          </div>
        </div>
      </div>

      {/* Dual Viewports */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="ui-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
              <Gamepad2 className="w-4 h-4 text-[var(--data-blue)]" />
              <span>You (click canvas or press Space to flap)</span>
            </span>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
                humanEngineRef.current.done
                  ? 'bg-[var(--status-danger-soft)] text-[var(--status-danger)]'
                  : 'bg-[var(--status-success-soft)] text-[var(--status-success)]'
              }`}
            >
              {humanEngineRef.current.done ? 'Terminated' : 'Active'}
            </span>
          </div>
          <GameCanvas
            engine={humanEngineRef.current}
            agentName="Human"
            agentColor="#38BDF8"
            showExplainability={false}
            aspectRatio="16:9"
            frameTick={frameTick}
            onCanvasClick={triggerHumanFlap}
            onRestart={() => resetMatch(seed)}
          />
        </div>

        <div className="ui-card p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-2 text-sm font-semibold text-[var(--text-primary)]">
              <Cpu className="w-4 h-4 text-[var(--accent-primary)]" />
              <span>AI — {aiMeta.name}</span>
            </span>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${
                aiEngineRef.current.done
                  ? 'bg-[var(--status-danger-soft)] text-[var(--status-danger)]'
                  : 'bg-[var(--status-success-soft)] text-[var(--status-success)]'
              }`}
            >
              {aiEngineRef.current.done ? 'Terminated' : 'Inference active'}
            </span>
          </div>
          <GameCanvas
            engine={aiEngineRef.current}
            decision={evaluateAgentDecision(
              aiOpponent,
              aiEngineRef.current.getState()
            )}
            agentName={aiMeta.shortName}
            agentColor="#F7D51D"
            showExplainability={true}
            aspectRatio="16:9"
            frameTick={frameTick}
          />
        </div>
      </div>
    </div>
  );
};
