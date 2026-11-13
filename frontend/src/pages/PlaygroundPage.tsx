import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  TrendingUp,
  Sliders,
  Network,
  Layers,
  ChevronDown,
  ChevronUp,
  ArrowUp,
  ArrowDown,
  ArrowRight,
} from 'lucide-react';
import { FlappyBirdEngine } from '../engine/flappyBird';
import {
  AGENT_CATALOG,
  AgentId,
  DecisionOutput,
  EXPERIMENTS_DATA,
  TimeMachineStageId,
  evaluateAgentDecision,
  preloadOnnxModel,
  MODELS_BUNDLE,
  getOnnxStatus,
} from '../engine/inference';
import {
  GameCanvas,
  EnvVisualTheme,
} from '../components/GameCanvas';
import { NeuralNetVisualizer } from '../components/NeuralNetVisualizer';
import { InfoTooltip } from '../components/InfoTooltip';

const CHECKPOINT_TIMELINE: {
  id: TimeMachineStageId;
  stageNum: number;
  shortLabel: string;
  stageTag: string;
  episode: number;
  title: string;
  subtitle: string;
  behaviorSummary: string;
  outcomeBadge: string;
  outcomeType: 'crash' | 'warn' | 'good' | 'master';
  color: string;
  epsilon: string;
  skills: {
    gravity: number;
    gapAlign: number;
    precision: number;
  };
  flightPath: string;
  endPoint: { x: number; y: number; crashed: boolean };
}[] = [
  {
    id: 'ep0',
    stageNum: 1,
    shortLabel: 'S1',
    stageTag: 'Untrained',
    episode: 0,
    title: 'Episode 0 — Untrained',
    subtitle: 'Random weights • No survival instinct',
    behaviorSummary:
      'Flaps erratically with random weights (ε = 100%) and nose-dives into the floor before reaching Pipe 1.',
    outcomeBadge: '× Crashes into floor',
    outcomeType: 'crash',
    color: '#EF4444',
    epsilon: '100%',
    skills: { gravity: 8, gapAlign: 4, precision: 2 },
    flightPath: 'M 12 22 Q 28 20, 44 36 L 62 46',
    endPoint: { x: 62, y: 46, crashed: true },
  },
  {
    id: 'ep100',
    stageNum: 2,
    shortLabel: 'S2',
    stageTag: 'Early learning',
    episode: 100,
    title: 'Episode 100 — Early learning',
    subtitle: 'Discovers gravity • Ignores pipe gaps',
    behaviorSummary:
      'Discovers that flapping prevents fatal ground crashes (+0.1 survival reward), staying airborne but flying into pipe walls.',
    outcomeBadge: '× Hits pipe wall',
    outcomeType: 'crash',
    color: '#F59E0B',
    epsilon: '65%',
    skills: { gravity: 72, gapAlign: 24, precision: 35 },
    flightPath: 'M 12 26 Q 28 40, 42 20 Q 58 38, 72 22 Q 84 36, 95 40',
    endPoint: { x: 95, y: 40, crashed: true },
  },
  {
    id: 'ep350',
    stageNum: 3,
    shortLabel: 'S3',
    stageTag: 'Intermediate',
    episode: 350,
    title: 'Episode 350 — Intermediate',
    subtitle: 'Aligns with gaps • Mistimes upper lips',
    behaviorSummary:
      'Uses Δx and Δy sensors to steer through pipe gaps (11.4 avg), but late panic-flaps still clip upper pipe lips.',
    outcomeBadge: '! Clips upper lip',
    outcomeType: 'warn',
    color: '#3B82F6',
    epsilon: '28%',
    skills: { gravity: 88, gapAlign: 68, precision: 72 },
    flightPath:
      'M 12 28 Q 44 36, 74 26 Q 103 24, 132 33 Q 160 36, 185 19',
    endPoint: { x: 185, y: 19, crashed: true },
  },
  {
    id: 'ep700',
    stageNum: 4,
    shortLabel: 'S4',
    stageTag: 'Advanced',
    episode: 700,
    title: 'Episode 700 — Advanced',
    subtitle: 'Horizon planning • Long pipe streaks',
    behaviorSummary:
      'Smoothly threads consecutive pipe heights (26.5 avg) by adjusting altitude immediately after clearing each pipe.',
    outcomeBadge: '✓ Multi-pipe streaks',
    outcomeType: 'good',
    color: '#06B6D4',
    epsilon: '8%',
    skills: { gravity: 95, gapAlign: 88, precision: 92 },
    flightPath:
      'M 12 26 Q 55 34, 103 26 Q 145 38, 193 32 Q 212 34, 228 28',
    endPoint: { x: 228, y: 28, crashed: false },
  },
  {
    id: 'ep1200',
    stageNum: 5,
    shortLabel: 'S5',
    stageTag: 'Fully trained',
    episode: 1200,
    title: 'Episode 1,200 — Fully trained',
    subtitle: 'Optimal lower-gap corridor mastery',
    behaviorSummary:
      'Hugs the lower-center safety corridor of every gap so its 62.5px flap arc never clips the upper pipe lip (42.0 avg).',
    outcomeBadge: '★ Optimal mastery',
    outcomeType: 'master',
    color: '#16A34A',
    epsilon: '2%',
    skills: { gravity: 99, gapAlign: 98, precision: 98 },
    flightPath:
      'M 12 28 C 48 32, 75 31, 103 29 C 138 30, 162 37, 193 35 C 208 35, 220 33, 230 31',
    endPoint: { x: 230, y: 31, crashed: false },
  },
];

const OBSERVATION_META: {
  key: string;
  label: string;
  min: number;
  max: number;
  signed: boolean;
  tooltip: string;
}[] = [
  {
    key: 'bird_y',
    label: 'Bird height',
    min: 0,
    max: 1,
    signed: false,
    tooltip:
      'Normalized vertical position y / GROUND_Y in [0, 1], where 0 is the ceiling and 1 is ground level (520px).',
  },
  {
    key: 'bird_vel',
    label: 'Vertical velocity',
    min: -1,
    max: 1,
    signed: true,
    tooltip:
      'Signed vertical velocity v_y / MAX_VEL in [−1, +1]. Negative values indicate upward climb after a flap; positive values indicate downward fall.',
  },
  {
    key: 'dx',
    label: 'Distance to next pipe',
    min: 0,
    max: 1,
    signed: false,
    tooltip:
      'Normalized horizontal distance Δx / WIDTH from the bird to the trailing edge of the upcoming pipe.',
  },
  {
    key: 'dy_upper',
    label: 'Distance to upper edge',
    min: -0.5,
    max: 0.5,
    signed: true,
    tooltip:
      'Signed vertical distance (y − gap_top) / GROUND_Y relative to the upper pipe lip. Positive when safely below the top pipe.',
  },
  {
    key: 'dy_lower',
    label: 'Distance to lower edge',
    min: -0.5,
    max: 0.5,
    signed: true,
    tooltip:
      'Signed vertical distance (gap_bottom − y) / GROUND_Y relative to the lower pipe lip. Positive when safely above the bottom pipe.',
  },
];

export const PlaygroundPage: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState<AgentId>('double_dqn');
  const [timeMachineActive, setTimeMachineActive] = useState<boolean>(false);
  const [timeMachineStage, setTimeMachineStage] =
    useState<TimeMachineStageId>('ep1200');
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(0.5);
  const [envTheme, setEnvTheme] = useState<EnvVisualTheme>('classic');
  const [seed] = useState<number>(101);
  const [liveEpisodeCount, setLiveEpisodeCount] = useState<number>(1);
  const [bestScore, setBestScore] = useState<number>(42);
  const [frameTick, setFrameTick] = useState<number>(0);
  const [activeBottomTab, setActiveBottomTab] = useState<
    'network' | 'architecture' | 'environment'
  >('network');
  const [showObsHistory, setShowObsHistory] = useState<boolean>(false);
  const [showProgressionGuide, setShowProgressionGuide] = useState<boolean>(false);
  const [mobileControlsOpen, setMobileControlsOpen] = useState<boolean>(false);
  const [autoEvolving, setAutoEvolving] = useState<boolean>(false);

  const engineRef = useRef<FlappyBirdEngine>(new FlappyBirdEngine(101));
  const ghostTrailRef = useRef<{ x: number; y: number }[]>([]);
  const obsHistoryRef = useRef<number[][]>([]);

  const [decision, setDecision] = useState<DecisionOutput>(() =>
    evaluateAgentDecision('double_dqn', engineRef.current.getState(), null)
  );

  const activeMeta =
    AGENT_CATALOG.find((a) => a.id === selectedAgent) ?? AGENT_CATALOG[5];

  useEffect(() => {
    const net = (MODELS_BUNDLE.networks as any)[selectedAgent];
    if (net?.onnx_url) {
      preloadOnnxModel(net.onnx_url);
    }
  }, [selectedAgent]);

  const handleRestart = useCallback(
    (newSeed?: number) => {
      const s = newSeed ?? seed;
      if (engineRef.current.trail.length > 2) {
        ghostTrailRef.current = engineRef.current.trail.map((pt) => ({
          x: pt.x,
          y: pt.y,
        }));
      }
      engineRef.current.reset(s);
      obsHistoryRef.current = [engineRef.current.getState()];
      const d = evaluateAgentDecision(
        selectedAgent,
        engineRef.current.getState(),
        timeMachineActive ? timeMachineStage : null
      );
      setDecision(d);
      setFrameTick((t) => t + 1);
    },
    [seed, selectedAgent, timeMachineActive, timeMachineStage]
  );

  useEffect(() => {
    handleRestart(seed);
  }, [selectedAgent, timeMachineActive, timeMachineStage, seed, handleRestart]);

  // Step a single simulation frame
  const stepSingleFrame = useCallback(() => {
    const eng = engineRef.current;
    if (eng.done) {
      if (eng.trail.length > 2) {
        ghostTrailRef.current = eng.trail.map((pt) => ({ x: pt.x, y: pt.y }));
      }
      const nextSeed = ((eng.currentSeed * 1664525 + 1013904223) >>> 0) % 9000 + 100;
      eng.reset(nextSeed);
      obsHistoryRef.current = [];
      setLiveEpisodeCount((e) => e + 1);
    }

    const currentState = eng.getState();
    const nextDecision = evaluateAgentDecision(
      selectedAgent,
      currentState,
      timeMachineActive ? timeMachineStage : null
    );
    eng.step(nextDecision.action);
    obsHistoryRef.current = [...obsHistoryRef.current.slice(-39), eng.getState()];

    if (eng.score > bestScore) {
      setBestScore(eng.score);
    }
    setDecision(nextDecision);
    setFrameTick((t) => t + 1);
  }, [selectedAgent, timeMachineActive, timeMachineStage, bestScore]);

  // Main simulation loop
  useEffect(() => {
    if (!isRunning) return;

    const intervalMs =
      speedMultiplier < 1 ? Math.round(16.67 / speedMultiplier) : 16.67;
    const stepsPerFrame = speedMultiplier >= 1 ? Math.round(speedMultiplier) : 1;

    const timer = window.setInterval(() => {
      const eng = engineRef.current;
      if (eng.done) {
        if (eng.trail.length > 2) {
          ghostTrailRef.current = eng.trail.map((pt) => ({ x: pt.x, y: pt.y }));
        }
        const nextSeed = ((eng.currentSeed * 1664525 + 1013904223) >>> 0) % 9000 + 100;
        eng.reset(nextSeed);
        obsHistoryRef.current = [];
        setLiveEpisodeCount((e) => e + 1);
      }

      let latestDecision = decision;
      for (let i = 0; i < stepsPerFrame; i++) {
        if (eng.done) break;
        const currentState = eng.getState();
        latestDecision = evaluateAgentDecision(
          selectedAgent,
          currentState,
          timeMachineActive ? timeMachineStage : null
        );
        eng.step(latestDecision.action);
        if (eng.score > bestScore) {
          setBestScore(eng.score);
        }
      }
      obsHistoryRef.current = [...obsHistoryRef.current.slice(-39), eng.getState()];
      setDecision(latestDecision);
      setFrameTick((t) => t + 1);
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [
    isRunning,
    speedMultiplier,
    selectedAgent,
    timeMachineActive,
    timeMachineStage,
    bestScore,
    decision,
  ]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (engineRef.current.done) {
          handleRestart(seed);
        } else {
          setIsRunning((r) => !r);
        }
      } else if (e.code === 'KeyR') {
        e.preventDefault();
        handleRestart(seed);
      } else if (e.code === 'Period' && !isRunning) {
        e.preventDefault();
        stepSingleFrame();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleRestart, seed, isRunning, stepSingleFrame]);

  // Auto-evolution tour across the 5 training checkpoints
  useEffect(() => {
    if (!autoEvolving) return;
    const timer = window.setInterval(() => {
      setTimeMachineStage((prev) => {
        const idx = CHECKPOINT_TIMELINE.findIndex((s) => s.id === prev);
        const nextIdx = (idx + 1) % CHECKPOINT_TIMELINE.length;
        return CHECKPOINT_TIMELINE[nextIdx].id;
      });
    }, 5500);
    return () => window.clearInterval(timer);
  }, [autoEvolving]);

  const currentState = engineRef.current.getState();
  const isCheckpointModel = timeMachineActive || selectedAgent === 'double_dqn';
  const effectiveStageId: TimeMachineStageId = timeMachineActive
    ? timeMachineStage
    : 'ep1200';
  const activeStageMeta =
    CHECKPOINT_TIMELINE.find((s) => s.id === effectiveStageId) ??
    CHECKPOINT_TIMELINE[4];
  const evalSummary = timeMachineActive
    ? EXPERIMENTS_DATA?.time_machine?.[timeMachineStage]?.evaluation ?? {
        mean_score: 0,
        std_score: 0,
        max_score: 0,
      }
    : EXPERIMENTS_DATA?.models?.[selectedAgent]?.evaluation ?? {
        mean_score: 0,
        std_score: 0,
        max_score: 0,
      };

  const onnxStatus = getOnnxStatus();
  const isProbPolicy =
    decision.policyType === 'policy_gradient' ||
    decision.policyType === 'stochastic';
  const qGlide = decision.qValues[0];
  const qFlap = decision.qValues[1];
  const qDiff = Math.abs(qGlide - qFlap);

  const maxAbsQ = Math.max(Math.abs(qGlide), Math.abs(qFlap), 1);
  const glideBarPct = Math.min(
    100,
    Math.max(8, ((qGlide + maxAbsQ) / (2 * maxAbsQ)) * 100)
  );
  const flapBarPct = Math.min(
    100,
    Math.max(8, ((qFlap + maxAbsQ) / (2 * maxAbsQ)) * 100)
  );

  return (
    <div className="space-y-3">
      {/* Mobile Collapsible Controls Toggle (< 1024px) */}
      <div className="flex lg:hidden items-center justify-between">
        <button
          type="button"
          onClick={() => setMobileControlsOpen((o) => !o)}
          className="ui-btn ui-btn-secondary w-full justify-between"
        >
          <span>
            Model & checkpoint controls (
            {timeMachineActive ? activeStageMeta.title : activeMeta.name})
          </span>
          {mobileControlsOpen ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Main Three-Column Playground Layout:
          Left (3 cols): Model & Checkpoint Timeline
          Center (6 cols): Game Viewport ON TOP + Neural Network Activity DIRECTLY BELOW (simultaneously visible!)
          Right (3 cols): Compact Performance Strip + Agent Decision + Live Observations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        {/* LEFT PANEL (3 cols): Agent Controls & Interactive Training Evolution */}
        <aside
          className={`lg:col-span-3 space-y-2.5 order-2 lg:order-1 ${
            mobileControlsOpen ? 'block' : 'hidden lg:block'
          }`}
        >
          {/* Agent Controls Card (Streamlined) */}
          <div className="ui-card p-3 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                  Agent model
                </h2>
                <InfoTooltip
                  label="Model architecture details"
                  text={`${activeMeta.name} (${activeMeta.architecture}, ${activeMeta.parameters.toLocaleString()} parameters): ${activeMeta.description}`}
                />
              </div>
              <span className="text-xs font-mono text-[var(--text-secondary)]">
                Level {activeMeta.level} • {activeMeta.parameters.toLocaleString()}p
              </span>
            </div>

            <select
              id="agent-model-select"
              aria-label="Select RL agent model"
              value={selectedAgent}
              onChange={(e) => {
                setSelectedAgent(e.target.value as AgentId);
                setTimeMachineActive(false);
                setAutoEvolving(false);
              }}
              className="w-full h-8 px-2.5 rounded-lg bg-[var(--bg-subtle)] border border-[var(--border-strong)] text-xs sm:text-sm font-medium text-[var(--text-primary)] cursor-pointer"
            >
              {AGENT_CATALOG.map((agent) => (
                <option key={agent.id} value={agent.id}>
                  Level {agent.level}: {agent.name}
                </option>
              ))}
            </select>

            {/* Expandable Agent Progression Story */}
            <div>
              <button
                type="button"
                onClick={() => setShowProgressionGuide((prev) => !prev)}
                className="w-full flex items-center justify-between text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
              >
                <span className="font-mono text-[11px] text-[var(--text-secondary)]">
                  {activeMeta.architecture}
                </span>
                <span className="inline-flex items-center gap-1 text-[var(--accent-primary)]">
                  <span>All 8 models</span>
                  {showProgressionGuide ? (
                    <ChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5" />
                  )}
                </span>
              </button>

              {showProgressionGuide && (
                <div className="mt-1.5 space-y-1 border-l-2 border-[var(--border-strong)] pl-2.5">
                  {AGENT_CATALOG.map((item) => {
                    const isCurrent = !timeMachineActive && selectedAgent === item.id;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => {
                          setSelectedAgent(item.id);
                          setTimeMachineActive(false);
                          setAutoEvolving(false);
                        }}
                        className={`w-full text-left px-2 py-1 rounded text-xs transition-colors cursor-pointer ${
                          isCurrent
                            ? 'bg-[var(--accent-primary-soft)] text-[var(--accent-primary)] font-semibold'
                            : 'hover:bg-[var(--bg-subtle)] text-[var(--text-secondary)]'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span>
                            L{item.level} • {item.shortName}
                          </span>
                          <span className="font-mono text-[11px]">
                            {item.parameters.toLocaleString()}p
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Creative Training Checkpoints & Behavioral Evolution Visualizer */}
          <div className="ui-card p-3 space-y-2">
            {/* Header + Watch Evolution Tour Button */}
            <div className="flex items-center justify-between gap-1">
              <div className="flex items-center gap-1.5">
                <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                  Training evolution
                </h2>
                <InfoTooltip
                  label="How AI learns across checkpoints"
                  text="Click any stage on the learning curve or press 'Watch AI learn' to observe how Double DQN evolves from random floor crashes at Episode 0 to optimal pipe-gap mastery at Episode 1,200."
                />
              </div>

              <button
                type="button"
                onClick={() => {
                  if (autoEvolving) {
                    setAutoEvolving(false);
                  } else {
                    setSelectedAgent('double_dqn');
                    setTimeMachineActive(true);
                    setTimeMachineStage('ep0');
                    setAutoEvolving(true);
                  }
                }}
                className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold transition-all cursor-pointer whitespace-nowrap ${
                  autoEvolving
                    ? 'bg-[#F97316] text-white shadow-sm ring-2 ring-[#F97316]/30'
                    : 'bg-[var(--accent-primary-soft)] text-[var(--accent-primary)] hover:bg-[#F97316] hover:text-white'
                }`}
              >
                {autoEvolving ? (
                  <>
                    <Pause className="w-3 h-3" />
                    <span>Evolving S{activeStageMeta.stageNum}/5</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3 h-3" />
                    <span>Watch AI learn</span>
                  </>
                )}
              </button>
            </div>

            {/* 1. Interactive Learning Curve + Exploration Decay Chart */}
            <div className="ui-subcard p-2 space-y-1">
              <div className="flex items-center justify-between text-[10px]">
                <span className="font-medium text-[var(--text-secondary)]">
                  Learning curve
                </span>
                <span className="font-mono font-semibold text-[var(--text-secondary)]">
                  Ep 0 → 1,200
                </span>
              </div>

              <svg
                viewBox="0 0 250 58"
                className="w-full h-13 overflow-visible select-none"
                role="img"
                aria-label="Interactive training progression curve across 5 checkpoints"
              >
                <defs>
                  <linearGradient id="ckptCurveGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#F97316" stopOpacity="0.28" />
                    <stop offset="100%" stopColor="#F97316" stopOpacity="0.02" />
                  </linearGradient>
                </defs>

                {/* Horizontal baseline */}
                <line
                  x1={18}
                  y1={45}
                  x2={232}
                  y2={45}
                  stroke="var(--border-subtle)"
                  strokeWidth={1}
                />

                {/* Dashed Exploration Rate (ε) decay line: 100% -> 65% -> 28% -> 8% -> 2% */}
                <polyline
                  fill="none"
                  stroke="var(--text-muted)"
                  strokeWidth={1.25}
                  strokeDasharray="3 3"
                  opacity={0.55}
                  points="22,12 74,22 126,34 178,41 230,44"
                />

                {/* Shaded area under reward/score curve */}
                <polygon
                  fill="url(#ckptCurveGrad)"
                  points="22,45 22,44 74,42 126,34 178,21 230,10 230,45"
                />

                {/* Solid Evaluation Score Curve: 0.3 -> 2.5 -> 11.4 -> 26.5 -> 42.0 */}
                <polyline
                  fill="none"
                  stroke="#F97316"
                  strokeWidth={2.25}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points="22,44 74,42 126,34 178,21 230,10"
                />

                {/* Interactive Checkpoint Nodes on Curve */}
                {CHECKPOINT_TIMELINE.map((stg, idx) => {
                  const xs = [22, 74, 126, 178, 230];
                  const ys = [44, 42, 34, 21, 10];
                  const cx = xs[idx];
                  const cy = ys[idx];
                  const isCurrent =
                    isCheckpointModel && effectiveStageId === stg.id;
                  const scoreVal =
                    EXPERIMENTS_DATA?.time_machine?.[stg.id]?.evaluation
                      ?.mean_score ?? 0;

                  return (
                    <g
                      key={stg.id}
                      className="cursor-pointer"
                      onClick={() => {
                        setSelectedAgent('double_dqn');
                        setTimeMachineActive(true);
                        setTimeMachineStage(stg.id);
                        setAutoEvolving(false);
                      }}
                    >
                      {isCurrent && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r={7.5}
                          fill={stg.color}
                          fillOpacity={0.22}
                        />
                      )}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isCurrent ? 4.2 : 3.2}
                        fill={isCurrent ? stg.color : 'var(--bg-card)'}
                        stroke={stg.color}
                        strokeWidth={2}
                      />
                      <text
                        x={cx}
                        y={cy - 6}
                        textAnchor="middle"
                        fontSize={8.5}
                        fontWeight={isCurrent ? 700 : 500}
                        fill={
                          isCurrent
                            ? 'var(--text-primary)'
                            : 'var(--text-secondary)'
                        }
                        fontFamily="JetBrains Mono, monospace"
                      >
                        {Number(scoreVal).toFixed(1)}
                      </text>
                      <text
                        x={cx}
                        y={55}
                        textAnchor="middle"
                        fontSize={8.5}
                        fontWeight={isCurrent ? 700 : 500}
                        fill={
                          isCurrent ? stg.color : 'var(--text-secondary)'
                        }
                      >
                        {stg.shortLabel}
                      </text>
                    </g>
                  );
                })}
              </svg>

              <div className="flex items-center justify-between text-[10px] text-[var(--text-secondary)]">
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 h-0.5 bg-[#F97316] inline-block rounded" />
                  <span>Avg pipes</span>
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 border-t border-dashed border-[var(--text-muted)] inline-block" />
                  <span>Exploration ε ({activeStageMeta.epsilon} random)</span>
                </span>
              </div>
            </div>

            {/* 2. Active Stage Behavioral Spotlight (Visual Flight Path + Competency Bars) */}
            <div
              className="rounded-xl p-2 border space-y-1.5 transition-colors"
              style={{
                borderColor: `${activeStageMeta.color}55`,
                backgroundColor: 'var(--bg-subtle)',
              }}
            >
              <div className="flex items-center justify-between gap-1 whitespace-nowrap">
                <span
                  className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wide text-white"
                  style={{ backgroundColor: activeStageMeta.color }}
                >
                  S{activeStageMeta.stageNum}/5 • {activeStageMeta.stageTag}
                </span>
                <span
                  className="text-[10px] font-mono font-semibold"
                  style={{ color: activeStageMeta.color }}
                >
                  {activeStageMeta.outcomeBadge}
                </span>
              </div>

              {/* Miniature Visual Flight Signature Arena */}
              <div className="rounded-lg bg-slate-900/95 border border-slate-800 p-1">
                <div className="flex items-center justify-between text-[10px] text-slate-300 px-1 mb-0.5 whitespace-nowrap">
                  <span className="font-medium">
                    Flight signature (Ep {activeStageMeta.episode})
                  </span>
                  <span className="font-mono text-slate-400">
                    ε = {activeStageMeta.epsilon}
                  </span>
                </div>
                <svg
                  viewBox="0 0 240 46"
                  className="w-full h-10 rounded overflow-hidden"
                  role="img"
                  aria-label={`Flight trajectory diagram for ${activeStageMeta.title}`}
                >
                  {/* Sky & Ground */}
                  <rect x={0} y={0} width={240} height={41} fill="#0F172A" />
                  <rect x={0} y={41} width={240} height={5} fill="#334155" />

                  {/* Pipe 1 (x=95, gap y=12..32) */}
                  <rect x={95} y={0} width={14} height={12} fill="#22C55E" opacity={0.75} />
                  <rect x={95} y={32} width={14} height={9} fill="#22C55E" opacity={0.75} />

                  {/* Pipe 2 (x=185, gap y=18..37) */}
                  <rect x={185} y={0} width={14} height={18} fill="#22C55E" opacity={0.75} />
                  <rect x={185} y={37} width={14} height={4} fill="#22C55E" opacity={0.75} />

                  {/* Target safety corridor highlight for advanced/master stages */}
                  {activeStageMeta.stageNum >= 4 && (
                    <>
                      <line
                        x1={92}
                        y1={27}
                        x2={112}
                        y2={27}
                        stroke="#4ADE80"
                        strokeWidth={1.2}
                        strokeDasharray="2 2"
                      />
                      <line
                        x1={182}
                        y1={33}
                        x2={202}
                        y2={33}
                        stroke="#4ADE80"
                        strokeWidth={1.2}
                        strokeDasharray="2 2"
                      />
                    </>
                  )}

                  {/* Characteristic Bird Trajectory */}
                  <path
                    d={activeStageMeta.flightPath}
                    fill="none"
                    stroke={activeStageMeta.color}
                    strokeWidth={2.2}
                    strokeDasharray="4 2"
                    strokeLinecap="round"
                  />

                  {/* Start point */}
                  <circle cx={12} cy={25} r={2.5} fill="#94A3B8" />

                  {/* End Bird / Crash Marker */}
                  <circle
                    cx={activeStageMeta.endPoint.x}
                    cy={Math.min(40, activeStageMeta.endPoint.y)}
                    r={3.8}
                    fill={activeStageMeta.endPoint.crashed ? '#EF4444' : '#FACC15'}
                    stroke="#FFFFFF"
                    strokeWidth={1.2}
                  />
                  {activeStageMeta.endPoint.crashed && (
                    <text
                      x={activeStageMeta.endPoint.x + 6}
                      y={Math.min(38, Math.max(10, activeStageMeta.endPoint.y + 2))}
                      fontSize={8}
                      fontWeight={700}
                      fill="#FCA5A5"
                    >
                      CRASH
                    </text>
                  )}
                </svg>
              </div>

              {/* Plain-English Behavioral Explanation */}
              <p className="text-[11px] text-[var(--text-primary)] leading-tight">
                {activeStageMeta.behaviorSummary}
              </p>

              {/* 3 Behavioral Skill Bars */}
              <div className="grid grid-cols-3 gap-1">
                {[
                  { label: 'Gravity', pct: activeStageMeta.skills.gravity },
                  { label: 'Gap aim', pct: activeStageMeta.skills.gapAlign },
                  { label: 'Precision', pct: activeStageMeta.skills.precision },
                ].map((skill) => (
                  <div
                    key={skill.label}
                    className="bg-[var(--bg-card)] px-1.5 py-1 rounded border border-[var(--border-subtle)]"
                  >
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="text-[var(--text-secondary)]">
                        {skill.label}
                      </span>
                      <span className="font-mono font-semibold text-[var(--text-primary)]">
                        {skill.pct}%
                      </span>
                    </div>
                    <div className="w-full h-1 bg-[var(--bg-subtle)] rounded-full overflow-hidden mt-0.5">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${skill.pct}%`,
                          backgroundColor: activeStageMeta.color,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 3. Connected 5-Stage Progression Stepper */}
            <div className="space-y-1">
              {CHECKPOINT_TIMELINE.map((stage) => {
                const isSelected =
                  isCheckpointModel && effectiveStageId === stage.id;
                const stageEval =
                  EXPERIMENTS_DATA?.time_machine?.[stage.id]?.evaluation
                    ?.mean_score ?? 0;

                return (
                  <button
                    key={stage.id}
                    type="button"
                    onClick={() => {
                      setSelectedAgent('double_dqn');
                      setTimeMachineActive(true);
                      setTimeMachineStage(stage.id);
                      setAutoEvolving(false);
                    }}
                    className={`w-full flex items-center justify-between gap-2 px-2 py-1 rounded-lg border text-left transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-[var(--accent-primary-soft)] border-[var(--accent-primary)] shadow-xs'
                        : 'bg-[var(--bg-card)] border-[var(--border-subtle)] hover:bg-[var(--bg-subtle)]'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span
                        className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white shrink-0"
                        style={{
                          backgroundColor: isSelected
                            ? stage.color
                            : 'var(--text-muted)',
                        }}
                      >
                        {stage.stageNum}
                      </span>
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-semibold text-[var(--text-primary)] truncate">
                            {stage.stageTag}
                          </span>
                          <span className="text-[10px] font-mono text-[var(--text-secondary)]">
                            Ep {stage.episode}
                          </span>
                        </div>
                        <div className="text-[10px] text-[var(--text-secondary)] truncate leading-tight">
                          {stage.subtitle}
                        </div>
                      </div>
                    </div>

                    <div className="text-right shrink-0">
                      <span className="text-xs font-mono font-bold text-[var(--text-primary)]">
                        {Number(stageEval).toFixed(1)}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Subtle Inference Ready Footer inside Left Panel */}
            <div className="pt-1 border-t border-[var(--border-subtle)] flex items-center justify-between text-[11px] text-[var(--text-secondary)]">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--status-success)]" />
                {onnxStatus.label}
              </span>
              <InfoTooltip
                label="Inference runtime"
                text="Models execute deterministically in-browser via ONNX Runtime Web / exported PyTorch weights."
              />
            </div>
          </div>
        </aside>

        {/* CENTER PANEL (6 cols): Full-Bleed Game Viewport ON TOP + Live Neural Network Activity DIRECTLY BELOW */}
        <section className="lg:col-span-6 space-y-2.5 order-1 lg:order-2">
          {/* TOP HALF: Game Canvas fills 100% of the box with all controls layered inside the canvas */}
          <GameCanvas
            engine={engineRef.current}
            decision={decision}
            agentName={activeMeta.shortName}
            agentColor="#F7D51D"
            showExplainability={false}
            envTheme={envTheme}
            fillBox={true}
            frameTick={frameTick}
            onRestart={() => handleRestart(seed)}
          >
            {/* Top-Left & Top-Right Floating HUD Layer Inside Game Canvas (kept compact so center score is clear) */}
            <div className="pointer-events-none absolute top-3 left-3 right-3 flex items-start justify-between gap-2">
              <div className="pointer-events-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/80 backdrop-blur-sm border border-white/15 text-xs font-semibold text-white shadow-sm">
                <span
                  className={`w-2 h-2 rounded-full ${
                    isRunning ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                <span>{activeMeta.name}</span>
                {isCheckpointModel && (
                  <span
                    className="ml-0.5 px-1.5 py-0.5 rounded text-[10px] font-bold text-white"
                    style={{ backgroundColor: activeStageMeta.color }}
                  >
                    S{activeStageMeta.stageNum}/5 • {activeStageMeta.stageTag}
                  </span>
                )}
              </div>

              <select
                aria-label="Environment visual theme"
                value={envTheme}
                onChange={(e) => setEnvTheme(e.target.value as EnvVisualTheme)}
                onClick={(e) => e.stopPropagation()}
                className="pointer-events-auto h-7 px-2.5 rounded-lg bg-slate-900/75 backdrop-blur-sm border border-white/15 text-xs font-medium text-white cursor-pointer shadow-sm outline-none"
              >
                <option value="classic" className="bg-slate-900 text-white">
                  Classic
                </option>
                <option value="sunset" className="bg-slate-900 text-white">
                  Sunset
                </option>
                <option value="night" className="bg-slate-900 text-white">
                  Night
                </option>
              </select>
            </div>

            {/* Bottom Floating Control Layer Inside Game Canvas */}
            <div
              onClick={(e) => e.stopPropagation()}
              className="absolute bottom-2.5 left-3 right-3 flex flex-wrap items-center justify-between gap-2 px-2.5 py-1.5 rounded-xl bg-slate-900/80 backdrop-blur-md border border-white/15 text-white shadow-md"
            >
              {/* Play / Pause, Single-Step, Reset */}
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setIsRunning((r) => !r)}
                  className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-lg bg-[#F97316] hover:bg-[#EA580C] text-white text-xs font-semibold transition-colors cursor-pointer"
                  title="Shortcut: Space"
                >
                  {isRunning ? (
                    <>
                      <Pause className="w-3.5 h-3.5" />
                      <span>Pause</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5" />
                      <span>Play</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setIsRunning(false);
                    stepSingleFrame();
                  }}
                  className="inline-flex items-center gap-1 h-7 px-2 rounded-lg bg-white/10 hover:bg-white/20 border border-white/15 text-white text-xs font-medium transition-colors cursor-pointer"
                  title="Advance one simulation step (Shortcut: .)"
                >
                  <StepForward className="w-3.5 h-3.5" />
                  <span>Step</span>
                </button>

                <button
                  type="button"
                  onClick={() => handleRestart(seed)}
                  className="inline-flex items-center gap-1 h-7 px-2 rounded-lg bg-white/10 hover:bg-white/20 border border-white/15 text-white text-xs font-medium transition-colors cursor-pointer"
                  title="Restart episode (Shortcut: R)"
                  aria-label="Restart episode"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Reset</span>
                </button>
              </div>

              {/* Speed Selector Inside Game Canvas */}
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] font-medium text-slate-300 hidden sm:inline">
                  Speed
                </span>
                <div
                  className="inline-flex items-center bg-slate-950/60 border border-white/10 rounded-lg p-0.5 gap-0.5"
                  role="group"
                  aria-label="Playback speed"
                >
                  {[0.25, 0.5, 1, 2, 4].map((spd) => (
                    <button
                      key={spd}
                      type="button"
                      onClick={() => setSpeedMultiplier(spd)}
                      className={`font-mono text-[11px] h-6 px-2 rounded-md transition-colors cursor-pointer ${
                        speedMultiplier === spd
                          ? 'bg-[#F97316] text-white font-semibold'
                          : 'text-slate-300 hover:text-white'
                      }`}
                    >
                      {spd}×
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </GameCanvas>

          {/* BOTTOM HALF OF CENTER PANEL: Live Neural Network Activity & Technical Tabs (Simultaneously Visible!) */}
          <div className="ui-card p-3 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-1.5">
              <div className="flex flex-wrap items-center gap-1" role="tablist">
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeBottomTab === 'network'}
                  onClick={() => setActiveBottomTab('network')}
                  className={`ui-btn h-7 px-2.5 text-xs ${
                    activeBottomTab === 'network'
                      ? 'ui-btn-primary'
                      : 'ui-btn-ghost'
                  }`}
                >
                  <Network className="w-3.5 h-3.5" />
                  <span>Neural network activity</span>
                </button>

                <button
                  type="button"
                  role="tab"
                  aria-selected={activeBottomTab === 'architecture'}
                  onClick={() => setActiveBottomTab('architecture')}
                  className={`ui-btn h-7 px-2.5 text-xs ${
                    activeBottomTab === 'architecture'
                      ? 'ui-btn-primary'
                      : 'ui-btn-ghost'
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>Architecture & progression</span>
                </button>

                <button
                  type="button"
                  role="tab"
                  aria-selected={activeBottomTab === 'environment'}
                  onClick={() => setActiveBottomTab('environment')}
                  className={`ui-btn h-7 px-2.5 text-xs ${
                    activeBottomTab === 'environment'
                      ? 'ui-btn-primary'
                      : 'ui-btn-ghost'
                  }`}
                >
                  <Sliders className="w-3.5 h-3.5" />
                  <span>Environment & rewards</span>
                </button>
              </div>
            </div>

            {activeBottomTab === 'network' && (
              <NeuralNetVisualizer
                state={currentState}
                decision={decision}
                agentName={
                  timeMachineActive
                    ? `Double DQN (${activeStageMeta.title})`
                    : activeMeta.name
                }
                architecture={activeMeta.architecture}
              />
            )}

            {activeBottomTab === 'architecture' && (
              <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
                <div className="md:col-span-5 ui-subcard p-3.5 space-y-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--accent-primary)]">
                    Level {activeMeta.level} • {activeMeta.name}
                  </span>
                  <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                    {activeMeta.description}
                  </p>
                  <div className="text-xs space-y-1 pt-1">
                    <div>
                      <strong className="text-[var(--text-primary)]">Improves: </strong>
                      <span className="text-[var(--text-secondary)]">
                        {activeMeta.improvementSummary}
                      </span>
                    </div>
                    <div>
                      <strong className="text-[var(--text-primary)]">Limitation: </strong>
                      <span className="text-[var(--text-secondary)]">
                        {activeMeta.limitations}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="md:col-span-7 grid grid-cols-2 gap-2">
                  {AGENT_CATALOG.map((item) => {
                    const mEval =
                      EXPERIMENTS_DATA?.models?.[item.id]?.evaluation?.mean_score ?? 0;
                    const isCurrent = selectedAgent === item.id && !timeMachineActive;
                    return (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => {
                          setSelectedAgent(item.id);
                          setTimeMachineActive(false);
                        }}
                        className={`text-left p-2.5 rounded-lg border transition-all cursor-pointer ${
                          isCurrent
                            ? 'bg-[var(--accent-primary-soft)] border-[var(--accent-primary)]'
                            : 'bg-[var(--bg-subtle)] border-[var(--border-subtle)] hover:border-[var(--border-strong)]'
                        }`}
                      >
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-semibold text-[var(--text-primary)]">
                            L{item.level} • {item.shortName}
                          </span>
                          <span className="font-mono font-semibold text-[var(--accent-primary)]">
                            {Number(mEval).toFixed(1)}
                          </span>
                        </div>
                        <p className="text-[11px] text-[var(--text-secondary)] mt-0.5 truncate">
                          {item.architecture}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {activeBottomTab === 'environment' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="ui-subcard p-3 space-y-1.5">
                  <h4 className="font-semibold text-[var(--text-primary)]">
                    Reward function
                  </h4>
                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between">
                      <span>Survive timestep</span>
                      <span className="text-[var(--data-blue)] font-semibold">+0.1</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Pass pipe gap</span>
                      <span className="text-[var(--status-success)] font-semibold">+1.0</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Collision / crash</span>
                      <span className="text-[var(--status-danger)] font-semibold">−10.0</span>
                    </div>
                  </div>
                </div>

                <div className="ui-subcard p-3 space-y-1.5">
                  <h4 className="font-semibold text-[var(--text-primary)]">
                    Kinematics & physics
                  </h4>
                  <ul className="space-y-1 text-[var(--text-secondary)] font-mono">
                    <li>• Glide (0): v_y ← min(10, v_y + 0.45)</li>
                    <li>• Flap (1): v_y ← −7.5 px/step</li>
                    <li>• Pipe gap: 120 px • Flap jump: 62.5 px</li>
                  </ul>
                </div>

                <div className="ui-subcard p-3 space-y-1.5">
                  <h4 className="font-semibold text-[var(--text-primary)]">
                    Camera invariance
                  </h4>
                  <p className="text-[var(--text-secondary)] leading-relaxed">
                    Display aspect ratios (16:9 / 3:4) and themes only scale the viewport camera;
                    underlying simulation physics and 5D observations remain strictly unchanged.
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* RIGHT PANEL (3 cols): Compact Performance Metrics + Agent Decision + Live Observations */}
        <aside className="lg:col-span-3 space-y-3.5 order-3">
          {/* 1. Compact Performance Summary Card (Replaces bulky top banner) */}
          <div className="ui-card p-4 space-y-2.5">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                Performance summary
              </h2>
              <span className="text-xs font-mono text-[var(--text-secondary)]">
                {timeMachineActive
                  ? `Checkpoint Ep ${activeStageMeta.episode}`
                  : `Eval run #${liveEpisodeCount}`}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="ui-subcard p-2.5">
                <span className="text-[11px] text-[var(--text-secondary)] block">
                  Current score
                </span>
                <span className="text-2xl font-bold font-mono text-[var(--text-primary)]">
                  {engineRef.current.score}
                </span>
              </div>
              <div className="ui-subcard p-2.5">
                <span className="text-[11px] text-[var(--text-secondary)] block">
                  Best score
                </span>
                <span className="text-2xl font-bold font-mono text-[var(--accent-primary)]">
                  {bestScore}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs pt-0.5">
              <div className="flex items-center justify-between px-1">
                <span className="text-[var(--text-secondary)] inline-flex items-center gap-1">
                  Eval avg
                  <InfoTooltip
                    label="Held-out evaluation average"
                    text="Mean pipes cleared across 10 held-out seeds with exploration disabled (ε = 0)."
                  />
                </span>
                <span className="font-mono font-semibold text-[var(--text-primary)]">
                  {Number(evalSummary.mean_score).toFixed(1)}
                </span>
              </div>
              <div className="flex items-center justify-between px-1">
                <span className="text-[var(--text-secondary)]">Reward</span>
                <span className="font-mono font-semibold text-[var(--text-primary)]">
                  {engineRef.current.cumulativeReward >= 0 ? '+' : ''}
                  {engineRef.current.cumulativeReward.toFixed(1)}
                </span>
              </div>
            </div>
          </div>

          {/* 2. Agent Decision Card (Signed Q-values for DQN, Probabilities for PPO) */}
          <div className="ui-card p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                  Agent decision
                </h2>
                <InfoTooltip
                  label="How agent decisions are computed"
                  text={
                    isProbPolicy
                      ? 'Policy-gradient and stochastic models output action selection probabilities π(a | s) over the discrete action space.'
                      : 'Value-based RL models estimate expected discounted return Q(s, a) for each action and greedily select argmax_a Q(s, a).'
                  }
                />
              </div>

              <span
                className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                  decision.action === 1
                    ? 'bg-[var(--accent-primary-soft)] text-[var(--accent-primary)]'
                    : 'bg-[var(--status-success-soft)] text-[var(--status-success)]'
                }`}
              >
                {decision.action === 1 ? (
                  <>
                    <ArrowUp className="w-3.5 h-3.5" />
                    <span>Flap selected</span>
                  </>
                ) : (
                  <>
                    <ArrowRight className="w-3.5 h-3.5" />
                    <span>Glide selected</span>
                  </>
                )}
              </span>
            </div>

            <div className="space-y-2">
              {/* Glide (Action 0) */}
              <div
                className={`p-2.5 rounded-lg border transition-colors ${
                  decision.action === 0
                    ? 'bg-[var(--status-success-soft)] border-[var(--status-success)]/40'
                    : 'bg-[var(--bg-subtle)] border-[var(--border-subtle)]'
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-semibold text-[var(--text-primary)] inline-flex items-center gap-1">
                    <ArrowRight className="w-3.5 h-3.5 text-[var(--status-success)]" />
                    Glide (0)
                  </span>
                  <span className="font-mono font-semibold text-[var(--text-primary)]">
                    {isProbPolicy
                      ? `π = ${(decision.actionProbs[0] * 100).toFixed(1)}%`
                      : `Q(s, glide) = ${
                          qGlide >= 0 ? `+${qGlide.toFixed(2)}` : qGlide.toFixed(2)
                        }`}
                  </span>
                </div>
                <div className="w-full h-1.5 bg-[var(--bg-card)] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[var(--status-success)] rounded-full transition-all duration-100"
                    style={{
                      width: `${
                        isProbPolicy ? decision.actionProbs[0] * 100 : glideBarPct
                      }%`,
                    }}
                  />
                </div>
              </div>

              {/* Flap (Action 1) */}
              <div
                className={`p-2.5 rounded-lg border transition-colors ${
                  decision.action === 1
                    ? 'bg-[var(--accent-primary-soft)] border-[var(--accent-primary)]/40'
                    : 'bg-[var(--bg-subtle)] border-[var(--border-subtle)]'
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-semibold text-[var(--text-primary)] inline-flex items-center gap-1">
                    <ArrowUp className="w-3.5 h-3.5 text-[var(--accent-primary)]" />
                    Flap (1)
                  </span>
                  <span className="font-mono font-semibold text-[var(--text-primary)]">
                    {isProbPolicy
                      ? `π = ${(decision.actionProbs[1] * 100).toFixed(1)}%`
                      : `Q(s, flap) = ${
                          qFlap >= 0 ? `+${qFlap.toFixed(2)}` : qFlap.toFixed(2)
                        }`}
                  </span>
                </div>
                <div className="w-full h-1.5 bg-[var(--bg-card)] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[var(--accent-primary)] rounded-full transition-all duration-100"
                    style={{
                      width: `${
                        isProbPolicy ? decision.actionProbs[1] * 100 : flapBarPct
                      }%`,
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-[var(--border-subtle)] space-y-1 text-xs">
              {!isProbPolicy && (
                <div className="flex items-center justify-between">
                  <span className="text-[var(--text-secondary)]">
                    Action value difference
                  </span>
                  <span className="font-mono font-semibold text-[var(--text-primary)]">
                    {qDiff.toFixed(2)}
                  </span>
                </div>
              )}
              {decision.stateValue !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-[var(--text-secondary)]">State value V(s)</span>
                  <span className="font-mono font-semibold text-[var(--text-primary)]">
                    {decision.stateValue >= 0
                      ? `+${decision.stateValue.toFixed(2)}`
                      : decision.stateValue.toFixed(2)}
                  </span>
                </div>
              )}
              <div className="flex items-center justify-between">
                <span className="text-[var(--text-secondary)]">Policy</span>
                <span className="font-medium text-[var(--text-primary)]">
                  {decision.policyLabel}
                </span>
              </div>
            </div>
          </div>

          {/* 3. Live Observations Card */}
          <div className="ui-card p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <h2 className="text-sm font-semibold text-[var(--text-primary)]">
                  Live observations
                </h2>
                <InfoTooltip
                  label="Five-dimensional state vector"
                  text="Normalized 5D environment observations passed into the model at every frame. Signed features use center-zero diverging bars."
                />
              </div>

              <button
                type="button"
                onClick={() => setShowObsHistory((h) => !h)}
                className="text-xs font-medium text-[var(--data-blue)] hover:underline inline-flex items-center gap-1 cursor-pointer"
              >
                <TrendingUp className="w-3.5 h-3.5" />
                <span>{showObsHistory ? 'Hide chart' : 'History'}</span>
              </button>
            </div>

            <div className="space-y-2.5">
              {OBSERVATION_META.map((obs, idx) => {
                const val = currentState[idx] ?? 0;
                const formatted =
                  obs.signed && val >= 0 ? `+${val.toFixed(3)}` : val.toFixed(3);

                return (
                  <div key={obs.key} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[var(--text-primary)] font-medium inline-flex items-center gap-1">
                        <span>{obs.label}</span>
                        <InfoTooltip label={obs.label} text={obs.tooltip} />
                      </span>
                      <span className="font-mono font-medium text-[var(--text-primary)] inline-flex items-center gap-1">
                        {obs.key === 'bird_vel' &&
                          (val < 0 ? (
                            <ArrowUp className="w-3 h-3 text-[var(--status-success)]" />
                          ) : (
                            <ArrowDown className="w-3 h-3 text-[var(--accent-primary)]" />
                          ))}
                        {formatted}
                      </span>
                    </div>

                    {obs.signed ? (
                      <div className="relative w-full h-1.5 bg-[var(--bg-subtle)] rounded-full overflow-hidden">
                        <div className="absolute left-1/2 top-0 bottom-0 w-[1.5px] bg-[var(--border-strong)] z-10" />
                        {val >= 0 ? (
                          <div
                            className="absolute left-1/2 top-0 bottom-0 bg-[var(--data-blue)] rounded-r-full transition-all duration-75"
                            style={{
                              width: `${Math.min(50, (val / obs.max) * 50)}%`,
                            }}
                          />
                        ) : (
                          <div
                            className="absolute top-0 bottom-0 bg-[var(--accent-primary)] rounded-l-full transition-all duration-75"
                            style={{
                              right: '50%',
                              width: `${Math.min(
                                50,
                                (Math.abs(val) / Math.abs(obs.min)) * 50
                              )}%`,
                            }}
                          />
                        )}
                      </div>
                    ) : (
                      <div className="w-full h-1.5 bg-[var(--bg-subtle)] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[var(--data-blue)] rounded-full transition-all duration-75"
                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(100, ((val - obs.min) / (obs.max - obs.min)) * 100)
                            )}%`,
                          }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {showObsHistory && (
              <div className="ui-subcard p-2.5 space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-[var(--text-secondary)]">
                  <span>Recent trace (40 steps)</span>
                  <span className="font-mono">Δy_lower & v_y</span>
                </div>
                <svg
                  viewBox="0 0 220 56"
                  className="w-full h-14 overflow-visible"
                  role="img"
                  aria-label="Recent observation history time series"
                >
                  <line
                    x1={0}
                    y1={28}
                    x2={220}
                    y2={28}
                    stroke="var(--border-strong)"
                    strokeDasharray="2 2"
                  />
                  {obsHistoryRef.current.length > 1 && (
                    <polyline
                      fill="none"
                      stroke="var(--data-blue)"
                      strokeWidth={2}
                      points={obsHistoryRef.current
                        .map((s, i) => {
                          const x = (i / 39) * 220;
                          const y = 28 - Math.max(-1, Math.min(1, s[4] * 2.5)) * 22;
                          return `${x},${y}`;
                        })
                        .join(' ')}
                    />
                  )}
                  {obsHistoryRef.current.length > 1 && (
                    <polyline
                      fill="none"
                      stroke="var(--accent-primary)"
                      strokeWidth={1.5}
                      strokeOpacity={0.8}
                      points={obsHistoryRef.current
                        .map((s, i) => {
                          const x = (i / 39) * 220;
                          const y = 28 - Math.max(-1, Math.min(1, s[1])) * 22;
                          return `${x},${y}`;
                        })
                        .join(' ')}
                    />
                  )}
                </svg>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
};
