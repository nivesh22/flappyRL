import * as ort from 'onnxruntime-web';
import modelsBundleRaw from '../data/modelsBundle.json';
import { Mulberry32 } from './prng';

export type AgentId =
  | 'random'
  | 'rule_based'
  | 'q_learning'
  | 'tiny_dqn'
  | 'simple_dqn'
  | 'deep_dqn'
  | 'double_dqn'
  | 'dueling_dqn'
  | 'ppo';

export type TimeMachineStageId = 'ep0' | 'ep100' | 'ep350' | 'ep700' | 'ep1200';

export type PolicyCategory = 'value_based' | 'policy_gradient' | 'heuristic' | 'stochastic';

export interface DecisionOutput {
  action: 0 | 1;
  qValues: [number, number]; // [Glide (0), Flap (1)]
  actionProbs: [number, number];
  hiddenActivations: number[][];
  layerWeights?: number[][][]; // [layerIdx][outIdx][inIdx]
  stateValue?: number;
  advantages?: [number, number];
  confidence: number;
  policyType: PolicyCategory;
  policyLabel: string;
}

interface LinearLayer {
  weight: number[][];
  bias: number[];
}

interface MLPWeights {
  type: 'mlp' | 'ppo';
  activation?: 'relu' | 'tanh';
  hidden_dims: number[];
  parameter_count: number;
  layers: LinearLayer[];
  critic_layers?: LinearLayer[];
  onnx_url?: string;
}

interface DuelingWeights {
  type: 'dueling';
  hidden_dims: number[];
  parameter_count: number;
  trunk: LinearLayer[];
  value_head: LinearLayer;
  advantage_head: LinearLayer;
  onnx_url?: string;
}

const bundle = modelsBundleRaw as unknown as {
  networks: Record<string, MLPWeights | DuelingWeights>;
  time_machine: Record<TimeMachineStageId, MLPWeights>;
  q_learning: {
    dy_bins: number;
    vel_bins: number;
    dx_bins: number;
    dy_edges: number[];
    vel_edges: number[];
    dx_edges: number[];
    q_table: number[][][][];
  };
  experiments: any;
};

export const EXPERIMENTS_DATA = bundle.experiments;
export const MODELS_BUNDLE = bundle;

export interface AgentMeta {
  id: AgentId;
  level: number;
  name: string;
  shortName: string;
  badge: string;
  color: string;
  architecture: string;
  parameters: number;
  policyType: PolicyCategory;
  description: string;
  improvementSummary: string;
  limitations: string;
}

export const AGENT_CATALOG: AgentMeta[] = [
  {
    id: 'random',
    level: 0,
    name: 'Random baseline',
    shortName: 'Random',
    badge: 'Level 0 • Baseline',
    color: '#64748B',
    architecture: 'Bernoulli(p = 0.12)',
    parameters: 1,
    policyType: 'stochastic',
    description:
      'Stochastic baseline that flaps with a 12% probability per timestep without observing state variables.',
    improvementSummary: 'Establishes the zero-learning reference floor under identical environment physics.',
    limitations: 'Ignores all sensor observations and fails before or at the first pipe.',
  },
  {
    id: 'rule_based',
    level: 1,
    name: 'Rule-based heuristic',
    shortName: 'Rule-based',
    badge: 'Level 1 • Heuristic',
    color: '#D97706',
    architecture: 'Threshold Heuristic (2 rules)',
    parameters: 2,
    policyType: 'heuristic',
    description:
      'Non-ML benchmark using hand-crafted threshold rules on vertical distance to the lower pipe edge and velocity.',
    improvementSummary: 'Demonstrates that domain heuristics can clear pipes without training.',
    limitations: 'Myopic 1-step threshold struggles with steep consecutive vertical pipe transitions.',
  },
  {
    id: 'q_learning',
    level: 2,
    name: 'Tabular Q-learning',
    shortName: 'Tabular Q',
    badge: 'Level 2 • Tabular RL',
    color: '#0D9488',
    architecture: 'Discretized Q-Table (14×8×6×2)',
    parameters: 1344,
    policyType: 'value_based',
    description:
      'Learns action values using a discretized lookup table over state bins without neural function approximation.',
    improvementSummary: 'First true RL agent; learns purely from reward signals (+0.1 step, +1.0 pipe, −10 crash).',
    limitations: 'Coarse state quantization causes aliasing near pipe edges (curse of dimensionality).',
  },
  {
    id: 'simple_dqn',
    level: 3,
    name: 'Simple DQN',
    shortName: 'Simple DQN',
    badge: 'Level 3 • Shallow NN',
    color: '#0284C7',
    architecture: 'MLP 5 → 32 → 2',
    parameters: 258,
    policyType: 'value_based',
    description:
      'Single hidden-layer Deep Q-Network with Experience Replay and Target Network over continuous 5D states.',
    improvementSummary: 'Replaces discrete state buckets with smooth continuous neural function approximation.',
    limitations: 'Standard max-operator in TD target introduces upward Q-value overestimation bias.',
  },
  {
    id: 'deep_dqn',
    level: 4,
    name: 'Deep DQN',
    shortName: 'Deep DQN',
    badge: 'Level 4 • Deep MLP',
    color: '#6366F1',
    architecture: 'MLP 5 → 128 → 64 → 32 → 2',
    parameters: 11170,
    policyType: 'value_based',
    description:
      'Three hidden-layer network with 43× more parameters than Simple DQN to explore network capacity.',
    improvementSummary: 'Higher representational capacity captures finer state boundaries across pipe transitions.',
    limitations: 'Larger parameter count requires more samples and remains susceptible to TD target noise.',
  },
  {
    id: 'double_dqn',
    level: 5,
    name: 'Double DQN',
    shortName: 'Double DQN',
    badge: 'Level 5 • Decoupled Target',
    color: '#F97316',
    architecture: 'MLP 5 → 64 → 32 → 2',
    parameters: 2530,
    policyType: 'value_based',
    description:
      'Decouples action selection (online network) from action evaluation (target network) to reduce overestimation bias.',
    improvementSummary: 'Eliminates max-operator overestimation, producing calibrated Q-values and stable pipe centering.',
    limitations: 'Evaluates full state-action values even in states where action choice has minimal immediate impact.',
  },
  {
    id: 'dueling_dqn',
    level: 6,
    name: 'Dueling Double DQN',
    shortName: 'Dueling DQN',
    badge: 'Level 6 • Value + Advantage',
    color: '#16A34A',
    architecture: '5 → 64 → 32 → [V(s) + A(s,a)]',
    parameters: 2563,
    policyType: 'value_based',
    description:
      'Separately estimates state value V(s) and action advantage A(s,a) before combining them into Q(s,a).',
    improvementSummary: 'Learns state safety V(s) rapidly across every timestep, accelerating convergence.',
    limitations: 'Slightly more complex head architecture with mean-advantage identifiability constraint.',
  },
  {
    id: 'ppo',
    level: 7,
    name: 'PPO (Actor-Critic)',
    shortName: 'PPO',
    badge: 'Level 7 • Policy Gradient',
    color: '#EC4899',
    architecture: 'Actor 5→64→32→2 | Critic 5→64→32→1',
    parameters: 5027,
    policyType: 'policy_gradient',
    description:
      'Proximal Policy Optimization directly optimizes a parameterized policy π(a|s) with clipped objective and GAE.',
    improvementSummary: 'Directly learns action selection probabilities without relying on Bellman Q-value maximization.',
    limitations: 'On-policy updates require fresh rollouts and separate actor and critic networks.',
  },
];

function applyLinear(input: number[], layer: LinearLayer): number[] {
  const out = new Array(layer.bias.length);
  for (let i = 0; i < layer.bias.length; i++) {
    let sum = layer.bias[i];
    const row = layer.weight[i];
    for (let j = 0; j < input.length; j++) {
      sum += row[j] * input[j];
    }
    out[i] = sum;
  }
  return out;
}

function relu(vec: number[]): number[] {
  return vec.map((v) => (v > 0 ? v : 0));
}

function tanhVec(vec: number[]): number[] {
  return vec.map((v) => Math.tanh(v));
}

function softmax(logits: [number, number]): [number, number] {
  const m = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - m);
  const e1 = Math.exp(logits[1] - m);
  const sum = e0 + e1;
  return [e0 / sum, e1 / sum];
}

function digitize(val: number, edges: number[]): number {
  let idx = 0;
  while (idx < edges.length && val >= edges[idx]) {
    idx++;
  }
  return idx;
}

function runMLPWeights(state: number[], net: MLPWeights): DecisionOutput {
  const activations: number[][] = [];
  let h = state;
  const isTanh = net.activation === 'tanh';
  const isPpo = net.type === 'ppo';

  for (let i = 0; i < net.layers.length - 1; i++) {
    h = applyLinear(h, net.layers[i]);
    h = isTanh ? tanhVec(h) : relu(h);
    activations.push(h);
  }

  const out = applyLinear(h, net.layers[net.layers.length - 1]);
  const qValues: [number, number] = [
    Math.round(out[0] * 100) / 100,
    Math.round(out[1] * 100) / 100,
  ];
  const probs = softmax(qValues);
  const action: 0 | 1 = out[1] > out[0] ? 1 : 0;

  let stateValue: number | undefined;
  if (net.critic_layers) {
    let ch = state;
    for (let i = 0; i < net.critic_layers.length - 1; i++) {
      ch = tanhVec(applyLinear(ch, net.critic_layers[i]));
    }
    stateValue =
      Math.round(applyLinear(ch, net.critic_layers[net.critic_layers.length - 1])[0] * 100) /
      100;
  }

  return {
    action,
    qValues,
    actionProbs: [Math.round(probs[0] * 1000) / 1000, Math.round(probs[1] * 1000) / 1000],
    hiddenActivations: activations,
    layerWeights: net.layers.map((l) => l.weight),
    stateValue,
    confidence: Math.abs(probs[1] - probs[0]),
    policyType: isPpo ? 'policy_gradient' : 'value_based',
    policyLabel: isPpo ? 'Stochastic Actor (argmax π)' : 'Greedy (argmax Q)',
  };
}

function runDuelingWeights(state: number[], net: DuelingWeights): DecisionOutput {
  const activations: number[][] = [];
  let h = state;
  for (let i = 0; i < net.trunk.length; i++) {
    h = relu(applyLinear(h, net.trunk[i]));
    activations.push(h);
  }
  const v = applyLinear(h, net.value_head)[0];
  const adv = applyLinear(h, net.advantage_head);
  const meanAdv = 0.5 * (adv[0] + adv[1]);
  const q0 = v + (adv[0] - meanAdv);
  const q1 = v + (adv[1] - meanAdv);
  const qValues: [number, number] = [
    Math.round(q0 * 100) / 100,
    Math.round(q1 * 100) / 100,
  ];
  const probs = softmax(qValues);

  return {
    action: q1 > q0 ? 1 : 0,
    qValues,
    actionProbs: [Math.round(probs[0] * 1000) / 1000, Math.round(probs[1] * 1000) / 1000],
    hiddenActivations: activations,
    layerWeights: [...net.trunk.map((l) => l.weight), net.advantage_head.weight],
    stateValue: Math.round(v * 100) / 100,
    advantages: [Math.round(adv[0] * 100) / 100, Math.round(adv[1] * 100) / 100],
    confidence: Math.abs(probs[1] - probs[0]),
    policyType: 'value_based',
    policyLabel: 'Greedy (V + A decomposition)',
  };
}

const onnxSessions: Record<string, ort.InferenceSession> = {};
let onnxBackendStatus: 'ready' | 'loading' | 'error' = 'ready';

export async function preloadOnnxModel(modelUrl: string): Promise<boolean> {
  if (onnxSessions[modelUrl]) return true;
  try {
    onnxBackendStatus = 'loading';
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });
    onnxSessions[modelUrl] = session;
    onnxBackendStatus = 'ready';
    return true;
  } catch {
    onnxBackendStatus = 'ready';
    return false;
  }
}

export function getOnnxStatus(): { state: 'ready' | 'loading' | 'error'; label: string } {
  if (onnxBackendStatus === 'loading') {
    return { state: 'loading', label: 'Loading ONNX weights…' };
  }
  return { state: 'ready', label: 'In-browser inference ready' };
}

const randomPrng = new Mulberry32(999);

export function evaluateAgentDecision(
  agentId: AgentId,
  state: number[],
  timeMachineStage?: TimeMachineStageId | null
): DecisionOutput {
  if (timeMachineStage && bundle.time_machine[timeMachineStage]) {
    const tmNet = bundle.time_machine[timeMachineStage];
    return runMLPWeights(state, tmNet);
  }

  if (agentId === 'random') {
    const r = randomPrng.nextFloat();
    const action: 0 | 1 = r < 0.12 ? 1 : 0;
    return {
      action,
      qValues: [0.0, 0.0],
      actionProbs: [0.88, 0.12],
      hiddenActivations: [[state[0], state[1], state[2], state[4]]],
      confidence: 0.12,
      policyType: 'stochastic',
      policyLabel: 'Bernoulli (p = 0.12)',
    };
  }

  if (agentId === 'rule_based') {
    const birdVel = state[1];
    const dyLower = state[4];
    const margin = 0.034;
    const velThreshold = 0.05;
    const shouldFlap = dyLower < margin && birdVel >= velThreshold;
    const urgency = (margin - dyLower) * 18.0;
    const q0 = Math.round((2.0 - urgency) * 100) / 100;
    const q1 = Math.round((2.0 + urgency) * 100) / 100;
    return {
      action: shouldFlap ? 1 : 0,
      qValues: [q0, q1],
      actionProbs: shouldFlap ? [0.0, 1.0] : [1.0, 0.0],
      hiddenActivations: [[dyLower < margin ? 1 : 0, birdVel >= velThreshold ? 1 : 0]],
      confidence: 1.0,
      policyType: 'heuristic',
      policyLabel: 'Threshold rule (deterministic)',
    };
  }

  if (agentId === 'q_learning') {
    const qt = bundle.q_learning;
    const bDy = digitize(state[4], qt.dy_edges);
    const bVel = digitize(state[1], qt.vel_edges);
    const bDx = digitize(state[2], qt.dx_edges);
    const pair = qt.q_table[bDy]?.[bVel]?.[bDx] ?? [0.1, 0.0];
    const q0 = Math.round(pair[0] * 100) / 100;
    const q1 = Math.round(pair[1] * 100) / 100;
    const probs = softmax([q0, q1]);
    return {
      action: q1 > q0 ? 1 : 0,
      qValues: [q0, q1],
      actionProbs: [Math.round(probs[0] * 1000) / 1000, Math.round(probs[1] * 1000) / 1000],
      hiddenActivations: [[bDy / qt.dy_bins, bVel / qt.vel_bins, bDx / qt.dx_bins]],
      confidence: Math.abs(probs[1] - probs[0]),
      policyType: 'value_based',
      policyLabel: 'Greedy lookup (argmax Q-table)',
    };
  }

  const net = bundle.networks[agentId];
  if (!net) {
    return runMLPWeights(state, bundle.networks['double_dqn'] as MLPWeights);
  }
  if (net.type === 'dueling') {
    return runDuelingWeights(state, net);
  }
  return runMLPWeights(state, net);
}
