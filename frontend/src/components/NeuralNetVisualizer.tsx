import React, { useEffect, useState } from 'react';
import { Pause, Play, Eye, Layers } from 'lucide-react';
import { DecisionOutput } from '../engine/inference';
import { InfoTooltip } from './InfoTooltip';

interface NeuralNetVisualizerProps {
  state: number[];
  decision: DecisionOutput;
  agentName: string;
  architecture?: string;
}

const OBSERVATION_NODES = [
  { key: 'y_pos', label: 'Bird height', symbol: 'y' },
  { key: 'y_vel', label: 'Vertical velocity', symbol: 'v_y' },
  { key: 'dx_pipe', label: 'Distance to next pipe', symbol: 'Δx' },
  { key: 'dy_top', label: 'Distance to upper edge', symbol: 'Δy_u' },
  { key: 'dy_bot', label: 'Distance to lower edge', symbol: 'Δy_l' },
];

export const NeuralNetVisualizer: React.FC<NeuralNetVisualizerProps> = ({
  state,
  decision,
  agentName,
  architecture,
}) => {
  const [viewMode, setViewMode] = useState<'activations' | 'weights'>('activations');
  const [densityMode, setDensityMode] = useState<'simplified' | 'full'>('simplified');
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [frozenSnapshot, setFrozenSnapshot] = useState<{
    state: number[];
    decision: DecisionOutput;
  }>({ state, decision });
  const [inspectedNode, setInspectedNode] = useState<{
    colIdx: number;
    rowIdx: number;
    origNeuronIdx: number;
    value: number;
    layerName: string;
  } | null>(null);

  useEffect(() => {
    if (!isPaused) {
      setFrozenSnapshot({ state, decision });
    }
  }, [state, decision, isPaused]);

  const activeState = isPaused ? frozenSnapshot.state : state;
  const activeDecision = isPaused ? frozenSnapshot.decision : decision;

  const maxNodesPerHidden = densityMode === 'simplified' ? 6 : 10;

  // Map hidden layers with original neuron indices preserved for accurate weight lookup & inspection
  const sampledHidden = activeDecision.hiddenActivations.map((layer) => {
    if (layer.length <= maxNodesPerHidden) {
      return layer.map((val, idx) => ({ val, origIdx: idx, total: layer.length }));
    }
    const step = layer.length / maxNodesPerHidden;
    return Array.from({ length: maxNodesPerHidden }, (_, i) => {
      const origIdx = Math.floor(i * step);
      return { val: layer[origIdx] ?? 0, origIdx, total: layer.length };
    });
  });

  const allColumns: { val: number; origIdx: number; total: number }[][] = [
    activeState.map((v, idx) => ({ val: v, origIdx: idx, total: activeState.length })),
    ...sampledHidden,
    activeDecision.qValues.map((v, idx) => ({ val: v, origIdx: idx, total: 2 })),
  ];

  const width = 940;
  const height = 225;
  const leftPad = 175;
  const rightPad = 195;
  const usableW = width - leftPad - rightPad;
  const colSpacing =
    allColumns.length > 1 ? usableW / (allColumns.length - 1) : usableW;

  const layerTitles = [
    'Observations (5)',
    ...activeDecision.hiddenActivations.map((l, idx) =>
      l.length > maxNodesPerHidden
        ? `Hidden ${idx + 1} (${maxNodesPerHidden} of ${l.length})`
        : `Hidden ${idx + 1} (${l.length})`
    ),
    activeDecision.policyType === 'policy_gradient' ? 'Policy Head (2)' : 'Action Values (2)',
  ];

  // Collect edges and filter in simplified mode to only show strongest connections
  interface EdgeItem {
    key: string;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    rawVal: number;
    absMag: number;
    isWinningOutput: boolean;
  }

  const allEdges: EdgeItem[] = [];
  for (let colIdx = 0; colIdx < allColumns.length - 1; colIdx++) {
    const col = allColumns[colIdx];
    const nextCol = allColumns[colIdx + 1];
    const x1 = leftPad + colIdx * colSpacing;
    const x2 = leftPad + (colIdx + 1) * colSpacing;
    const weightMat = activeDecision.layerWeights?.[colIdx];

    col.forEach((node1, i) => {
      const y1 = 26 + ((i + 0.5) / col.length) * (height - 34);
      nextCol.forEach((node2, j) => {
        const y2 = 26 + ((j + 0.5) / nextCol.length) * (height - 34);
        const wVal = weightMat?.[node2.origIdx]?.[node1.origIdx] ?? 0.35;
        const rawVal =
          viewMode === 'weights' ? wVal : node1.val * node2.val * (wVal >= 0 ? 1 : -1);
        const absMag = Math.abs(rawVal);
        const isWinningOutput =
          colIdx === allColumns.length - 2 && j === activeDecision.action;

        allEdges.push({
          key: `edge-${colIdx}-${i}-${j}`,
          x1,
          y1,
          x2,
          y2,
          rawVal,
          absMag,
          isWinningOutput,
        });
      });
    });
  }

  // In simplified mode, keep top 35% strongest edges or winning output edges
  const sortedMags = [...allEdges].map((e) => e.absMag).sort((a, b) => b - a);
  const threshold =
    densityMode === 'simplified'
      ? sortedMags[Math.min(sortedMags.length - 1, Math.floor(sortedMags.length * 0.35))] ?? 0.02
      : 0;

  const visibleEdges = allEdges.filter(
    (e) => densityMode === 'full' || e.isWinningOutput || e.absMag >= threshold
  );

  return (
    <div className="space-y-2">
      {/* Compact Controls & Metadata Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-[var(--bg-subtle)] text-[var(--text-secondary)] border border-[var(--border-subtle)]">
            {agentName}
          </span>
          {architecture && (
            <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-[var(--bg-subtle)] text-[var(--text-secondary)] border border-[var(--border-subtle)]">
              {architecture}
            </span>
          )}
          <InfoTooltip
            label="About neural network visualization"
            text="Displays input observations, representative hidden-layer activations (or static layer weights), and output action values. Highlighted edges reflect signal magnitude rather than causal attribution."
          />
        </div>

        {/* Mode & Density Controls */}
        <div className="flex flex-wrap items-center gap-1.5">
          {/* Activations vs Weights */}
          <div className="ui-segmented" role="group" aria-label="Encoding mode">
            <button
              type="button"
              onClick={() => setViewMode('activations')}
              className={`ui-segmented-item text-xs h-6 px-2 ${
                viewMode === 'activations' ? 'ui-segmented-item-active' : ''
              }`}
            >
              <Eye className="w-3 h-3" />
              Activations
            </button>
            <button
              type="button"
              onClick={() => setViewMode('weights')}
              className={`ui-segmented-item text-xs h-6 px-2 ${
                viewMode === 'weights' ? 'ui-segmented-item-active' : ''
              }`}
            >
              <Layers className="w-3 h-3" />
              Weights
            </button>
          </div>

          {/* Simplified vs Full Network */}
          <div className="ui-segmented" role="group" aria-label="Connection density">
            <button
              type="button"
              onClick={() => setDensityMode('simplified')}
              className={`ui-segmented-item text-xs h-6 px-2 ${
                densityMode === 'simplified' ? 'ui-segmented-item-active' : ''
              }`}
            >
              Top connections
            </button>
            <button
              type="button"
              onClick={() => setDensityMode('full')}
              className={`ui-segmented-item text-xs h-6 px-2 ${
                densityMode === 'full' ? 'ui-segmented-item-active' : ''
              }`}
            >
              Full view
            </button>
          </div>

          {/* Independent Pause Toggle */}
          <button
            type="button"
            onClick={() => setIsPaused((p) => !p)}
            className="ui-btn ui-btn-secondary text-xs h-7 px-2"
            title="Freeze network inspection independently of game playback"
          >
            {isPaused ? (
              <>
                <Play className="w-3 h-3 text-[var(--status-success)]" />
                <span>Resume</span>
              </>
            ) : (
              <>
                <Pause className="w-3 h-3" />
                <span>Freeze</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Theme-Matched SVG Diagram */}
      <div className="ui-subcard px-2.5 py-1.5">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto max-h-[25vh] select-none"
          role="img"
          aria-label={`Neural network ${viewMode} diagram for ${agentName}`}
        >
          {/* Layer Column Titles */}
          {allColumns.map((_, colIdx) => {
            const x = leftPad + colIdx * colSpacing;
            return (
              <text
                key={`col-title-${colIdx}`}
                x={x}
                y={15}
                textAnchor="middle"
                fill="var(--text-secondary)"
                className="text-[11.5px] font-sans font-semibold"
              >
                {layerTitles[colIdx]}
              </text>
            );
          })}

          {/* Synaptic Connections */}
          {visibleEdges.map((edge) => {
            const normMag = Math.min(1, edge.absMag * 1.8 + 0.12);
            let stroke = 'var(--text-muted)';
            if (edge.isWinningOutput) {
              stroke =
                activeDecision.action === 1
                  ? 'var(--accent-primary)'
                  : 'var(--status-success)';
            } else if (edge.rawVal >= 0) {
              stroke = 'var(--data-blue)';
            } else {
              stroke = '#F59E0B';
            }

            return (
              <line
                key={edge.key}
                x1={edge.x1}
                y1={edge.y1}
                x2={edge.x2}
                y2={edge.y2}
                stroke={stroke}
                strokeOpacity={
                  edge.isWinningOutput ? Math.max(0.55, normMag * 0.9) : normMag * 0.45
                }
                strokeWidth={
                  edge.isWinningOutput ? 1.8 + normMag * 1.8 : 0.8 + normMag * 1.4
                }
                className="transition-all duration-150"
              />
            );
          })}

          {/* Neurons */}
          {allColumns.map((col, colIdx) => {
            const x = leftPad + colIdx * colSpacing;
            const isInput = colIdx === 0;
            const isOutput = colIdx === allColumns.length - 1;

            return col.map((node, rowIdx) => {
              const y = 26 + ((rowIdx + 0.5) / col.length) * (height - 34);
              const val = node.val;
              const norm = Math.min(1, Math.abs(val));
              const selectedOutput = isOutput && rowIdx === activeDecision.action;
              const isInspected =
                inspectedNode?.colIdx === colIdx && inspectedNode?.rowIdx === rowIdx;

              let fill = 'var(--bg-card)';
              let stroke = 'var(--border-strong)';
              let textFill = 'var(--text-primary)';

              if (isOutput) {
                if (selectedOutput) {
                  fill =
                    rowIdx === 1 ? 'var(--accent-primary)' : 'var(--status-success)';
                  stroke = fill;
                  textFill = '#FFFFFF';
                }
              } else if (val > 0.02) {
                fill = `rgba(2, 132, 199, ${0.15 + norm * 0.7})`;
                stroke = 'var(--data-blue)';
                if (norm > 0.55) textFill = '#FFFFFF';
              } else if (val < -0.02) {
                fill = `rgba(249, 115, 22, ${0.15 + norm * 0.7})`;
                stroke = 'var(--accent-primary)';
                if (norm > 0.55) textFill = '#FFFFFF';
              }

              const radius = isOutput ? 16 : isInput ? 12.5 : 11.5;

              return (
                <g
                  key={`node-${colIdx}-${rowIdx}`}
                  className="cursor-pointer"
                  onMouseEnter={() =>
                    setInspectedNode({
                      colIdx,
                      rowIdx,
                      origNeuronIdx: node.origIdx,
                      value: val,
                      layerName: layerTitles[colIdx],
                    })
                  }
                  onClick={() =>
                    setInspectedNode({
                      colIdx,
                      rowIdx,
                      origNeuronIdx: node.origIdx,
                      value: val,
                      layerName: layerTitles[colIdx],
                    })
                  }
                >
                  {(selectedOutput || isInspected) && (
                    <circle
                      cx={x}
                      cy={y}
                      r={radius + 5}
                      fill="none"
                      stroke={
                        selectedOutput
                          ? rowIdx === 1
                            ? 'var(--accent-primary)'
                            : 'var(--status-success)'
                          : 'var(--data-blue)'
                      }
                      strokeWidth={2}
                      strokeOpacity={0.65}
                    />
                  )}

                  <circle
                    cx={x}
                    cy={y}
                    r={radius}
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={selectedOutput || isInspected ? 2.5 : 1.5}
                    className="transition-colors duration-150"
                  />

                  <text
                    x={x}
                    y={y + 3.5}
                    textAnchor="middle"
                    fill={textFill}
                    className="text-[10px] font-mono font-semibold pointer-events-none"
                  >
                    {val.toFixed(isOutput ? 1 : 2)}
                  </text>

                  {/* Left Input Observation Labels */}
                  {isInput && (
                    <g>
                      <text
                        x={x - 22}
                        y={y - 2}
                        textAnchor="end"
                        fill="var(--text-primary)"
                        className="text-[12px] font-sans font-medium"
                      >
                        {OBSERVATION_NODES[rowIdx]?.label}
                      </text>
                      <text
                        x={x - 22}
                        y={y + 12}
                        textAnchor="end"
                        fill="var(--text-secondary)"
                        className="text-[11px] font-mono"
                      >
                        {OBSERVATION_NODES[rowIdx]?.symbol} ={' '}
                        {val >= 0 ? `+${val.toFixed(3)}` : val.toFixed(3)}
                      </text>
                    </g>
                  )}

                  {/* Right Output Action Summary */}
                  {isOutput && (
                    <g>
                      <rect
                        x={x + 26}
                        y={y - 20}
                        width={136}
                        height={40}
                        rx={8}
                        fill="var(--bg-card)"
                        stroke={
                          selectedOutput
                            ? rowIdx === 1
                              ? 'var(--accent-primary)'
                              : 'var(--status-success)'
                            : 'var(--border-subtle)'
                        }
                        strokeWidth={selectedOutput ? 2 : 1}
                      />
                      <text
                        x={x + 36}
                        y={y - 3}
                        fill="var(--text-primary)"
                        className="text-[12px] font-sans font-semibold"
                      >
                        {rowIdx === 0 ? 'Glide (action 0)' : 'Flap (action 1)'}
                      </text>
                      <text
                        x={x + 36}
                        y={y + 12}
                        fill="var(--text-secondary)"
                        className="text-[11px] font-mono"
                      >
                        {activeDecision.policyType === 'policy_gradient' ||
                        activeDecision.policyType === 'stochastic'
                          ? `π = ${(activeDecision.actionProbs[rowIdx] * 100).toFixed(1)}%`
                          : `Q = ${val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}`}
                      </text>
                    </g>
                  )}
                </g>
              );
            });
          })}
        </svg>
      </div>

      {/* Bottom Legend, Neuron Inspector & Sampling Note */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-[var(--text-secondary)]">
        {/* Visual Encoding Legend */}
        <div className="flex flex-wrap items-center gap-4">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#0284C7]/70 border border-[#0284C7]" />
            <span>Positive activation / weight</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#F97316]/70 border border-[#F97316]" />
            <span>Negative value / Flap action</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#16A34A] border border-[#16A34A]" />
            <span>Selected Glide action</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-4 h-0.5 bg-[var(--text-secondary)]" />
            <span>Edge thickness = {viewMode === 'weights' ? '|W_ij|' : '|a_i · a_j|'}</span>
          </span>
        </div>

        {/* Inspected Neuron Readout or Sampling Disclaimer */}
        {inspectedNode ? (
          <div className="font-mono text-xs px-2.5 py-1 rounded bg-[var(--bg-subtle)] border border-[var(--border-subtle)] text-[var(--text-primary)]">
            {inspectedNode.layerName} • Neuron #{inspectedNode.origNeuronIdx}:{' '}
            <strong>
              {inspectedNode.value >= 0
                ? `+${inspectedNode.value.toFixed(4)}`
                : inspectedNode.value.toFixed(4)}
            </strong>
          </div>
        ) : (
          <span>
            Illustrative view: hidden layers show representative sampled nodes. Activations do not
            alone establish causal feature importance.
          </span>
        )}
      </div>
    </div>
  );
};
