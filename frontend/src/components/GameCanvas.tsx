import React, { useEffect, useRef, useState } from 'react';
import { FlappyBirdEngine, GAME_CONSTANTS } from '../engine/flappyBird';
import { DecisionOutput } from '../engine/inference';

export type EnvVisualTheme = 'classic' | 'sunset' | 'night';
export type CanvasAspectRatio = '16:9' | '3:4';

interface GameCanvasProps {
  engine: FlappyBirdEngine;
  decision?: DecisionOutput;
  agentName: string;
  agentColor?: string;
  showExplainability?: boolean;
  showTrail?: boolean;
  ghostTrail?: { x: number; y: number }[];
  envTheme?: EnvVisualTheme;
  aspectRatio?: CanvasAspectRatio;
  compact?: boolean;
  fillBox?: boolean;
  frameTick: number;
  onCanvasClick?: () => void;
  onRestart?: () => void;
  children?: React.ReactNode;
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

const THEME_PALETTES: Record<
  EnvVisualTheme,
  {
    skyTop: string;
    skyMid: string;
    skyBot: string;
    cloud: string;
    cityFill: string;
    cityStroke: string;
    cityWin: string;
    bushLight: string;
    bushDark: string;
  }
> = {
  classic: {
    skyTop: '#4EC0CA',
    skyMid: '#70C5CE',
    skyBot: '#BAE6FD',
    cloud: '#ECFEFF',
    cityFill: '#B8E9C8',
    cityStroke: '#84C69B',
    cityWin: '#84C69B',
    bushLight: '#83D442',
    bushDark: '#68B828',
  },
  sunset: {
    skyTop: '#4C1D95',
    skyMid: '#C2410C',
    skyBot: '#FDBA74',
    cloud: '#FED7AA',
    cityFill: '#7C2D12',
    cityStroke: '#431407',
    cityWin: '#FDE047',
    bushLight: '#4D7C0F',
    bushDark: '#365314',
  },
  night: {
    skyTop: '#090D16',
    skyMid: '#1E293B',
    skyBot: '#334155',
    cloud: '#475569',
    cityFill: '#0F172A',
    cityStroke: '#334155',
    cityWin: '#38BDF8',
    bushLight: '#15803D',
    bushDark: '#14532D',
  },
};

export const GameCanvas: React.FC<GameCanvasProps> = ({
  engine,
  agentName,
  agentColor = '#F7D51D',
  showExplainability = false,
  showTrail = true,
  ghostTrail,
  envTheme = 'classic',
  aspectRatio = '16:9',
  compact = false,
  fillBox = false,
  frameTick,
  onCanvasClick,
  onRestart,
  children,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [boxSize, setBoxSize] = useState<{ w: number; h: number }>({
    w: compact ? 400 : aspectRatio === '16:9' ? 960 : 450,
    h: compact ? 520 : aspectRatio === '16:9' ? 540 : 600,
  });

  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 20 && height > 20) {
          setBoxSize({ w: Math.round(width), h: Math.round(height) });
        }
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const logicalW = boxSize.w;
  const logicalH = boxSize.h;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;
    if (
      canvas.width !== Math.round(logicalW * dpr) ||
      canvas.height !== Math.round(logicalH * dpr)
    ) {
      canvas.width = Math.round(logicalW * dpr);
      canvas.height = Math.round(logicalH * dpr);
    }

    // Separate display camera from simulation coordinates (simulation height is always 600)
    const simH = GAME_CONSTANTS.HEIGHT;
    const scale = logicalH / simH;
    const simW = logicalW / scale;
    const G_Y = GAME_CONSTANTS.GROUND_Y;
    const camOffsetX = simW > 650 && !compact ? 55 : 0;

    ctx.save();
    ctx.scale(dpr * scale, dpr * scale);

    const pal = THEME_PALETTES[envTheme];

    // 1. Sky Backdrop
    const skyGrad = ctx.createLinearGradient(0, 0, 0, G_Y);
    skyGrad.addColorStop(0, pal.skyTop);
    skyGrad.addColorStop(0.7, pal.skyMid);
    skyGrad.addColorStop(1, pal.skyBot);
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, simW, G_Y);

    // Night stars if night theme
    if (envTheme === 'night') {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      for (let s = 0; s < 32; s++) {
        const sx = (s * 97 + 31) % Math.max(1, Math.ceil(simW));
        const sy = (s * 53 + 19) % (G_Y - 160);
        ctx.fillRect(sx, sy, 2, 2);
      }
    }

    // 2. Parallax Clouds & City Skyline
    const tileW = 400;
    const cloudOffset = (engine.steps * 0.35) % tileW;
    ctx.fillStyle = pal.cloud;
    const numTiles = Math.ceil(simW / tileW) + 1;
    for (let i = -1; i <= numTiles; i++) {
      const baseX = i * tileW - cloudOffset;
      const clouds = [
        [baseX + 40, G_Y - 95, 34],
        [baseX + 75, G_Y - 102, 42],
        [baseX + 115, G_Y - 92, 32],
        [baseX + 210, G_Y - 90, 36],
        [baseX + 250, G_Y - 104, 46],
        [baseX + 295, G_Y - 92, 34],
        [baseX + 365, G_Y - 98, 38],
      ];
      for (const [cx, cy, cr] of clouds) {
        ctx.beginPath();
        ctx.arc(cx, cy, cr, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillRect(baseX, G_Y - 92, tileW, 92);
    }

    // City Silhouette
    const cityOffset = (engine.steps * 0.6) % 200;
    ctx.fillStyle = pal.cityFill;
    ctx.strokeStyle = pal.cityStroke;
    ctx.lineWidth = 1.5;
    const bldWidths = [28, 38, 24, 44, 32, 34];
    const bldHeights = [68, 92, 54, 82, 62, 74];
    const cityLoops = Math.ceil(simW / 200) + 1;
    for (let cycle = -1; cycle <= cityLoops; cycle++) {
      let curX = cycle * 200 - cityOffset;
      for (let b = 0; b < bldWidths.length; b++) {
        const bw = bldWidths[b];
        const bh = bldHeights[b];
        ctx.fillRect(curX, G_Y - bh, bw, bh);
        ctx.strokeRect(curX, G_Y - bh, bw, bh);
        ctx.fillStyle = pal.cityWin;
        for (let wy = G_Y - bh + 8; wy < G_Y - 24; wy += 12) {
          ctx.fillRect(curX + 6, wy, 5, 6);
          if (bw > 30) ctx.fillRect(curX + 16, wy, 5, 6);
        }
        ctx.fillStyle = pal.cityFill;
        curX += bw;
      }
    }

    // Horizon Bushes
    const bushOffset = (engine.steps * 1.1) % 120;
    ctx.fillStyle = pal.bushLight;
    for (let x = -120; x < simW + 120; x += 30) {
      ctx.beginPath();
      ctx.arc(x - bushOffset, G_Y - 6, 22, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = pal.bushDark;
    ctx.fillRect(0, G_Y - 10, simW, 10);

    // Apply camera horizontal offset for world objects
    ctx.save();
    ctx.translate(camOffsetX, 0);

    // 3. Optional Ghost Replay Trail
    if (ghostTrail && ghostTrail.length > 1) {
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < ghostTrail.length; i++) {
        const pt = ghostTrail[i];
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.restore();
    }

    // 4. Active Flight Trajectory Trail
    if (showTrail && engine.trail.length > 1) {
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < engine.trail.length; i++) {
        const pt = engine.trail[i];
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([5, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      for (const pt of engine.trail) {
        if (pt.flapped) {
          ctx.fillStyle = '#F97316';
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 3.8, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    // 5. Authentic 3D-Shaded Flappy Bird Green Pipes
    const nextPipe = engine.getNextPipe();
    for (const pipe of engine.pipes) {
      const pw = GAME_CONSTANTS.PIPE_WIDTH;
      const px = pipe.x;
      if (px + pw + camOffsetX < -60 || px + camOffsetX > simW + 60) continue;

      const drawFlappyPipeSegment = (x: number, y: number, w: number, h: number) => {
        if (h <= 0) return;
        ctx.fillStyle = '#73BF2E';
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = '#9CE659';
        ctx.fillRect(x + 5, y, 8, h);
        ctx.fillRect(x + 16, y, 3, h);
        ctx.fillStyle = '#558022';
        ctx.fillRect(x + w - 12, y, 12, h);
        ctx.fillStyle = '#406318';
        ctx.fillRect(x + w - 5, y, 5, h);
        ctx.strokeStyle = '#26380F';
        ctx.lineWidth = 2.2;
        ctx.strokeRect(x, y, w, h);
      };

      const capH = 22;
      drawFlappyPipeSegment(px, -2, pw, Math.max(0, pipe.gapTopY - capH + 2));
      drawFlappyPipeSegment(px - 4, pipe.gapTopY - capH, pw + 8, capH);

      const lowerBodyY = pipe.gapBottomY + capH;
      const lowerBodyH = G_Y - lowerBodyY;
      drawFlappyPipeSegment(px, lowerBodyY, pw, Math.max(0, lowerBodyH));
      drawFlappyPipeSegment(px - 4, pipe.gapBottomY, pw + 8, capH);
    }

    // 6. Optional Debug Overlay
    if (showExplainability && nextPipe) {
      const bx = GAME_CONSTANTS.BIRD_X;
      const by = engine.birdY;
      const targetX = nextPipe.x + GAME_CONSTANTS.PIPE_WIDTH * 0.5;
      const targetY = nextPipe.gapCenterY;

      ctx.save();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
      ctx.strokeStyle = '#0284C7';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.fillRect(
        nextPipe.x,
        nextPipe.gapTopY,
        GAME_CONSTANTS.PIPE_WIDTH,
        GAME_CONSTANTS.PIPE_GAP
      );
      ctx.strokeRect(
        nextPipe.x,
        nextPipe.gapTopY,
        GAME_CONSTANTS.PIPE_WIDTH,
        GAME_CONSTANTS.PIPE_GAP
      );

      const triggerY = nextPipe.gapBottomY - 0.048 * G_Y;
      ctx.strokeStyle = '#F97316';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(nextPipe.x - 10, triggerY);
      ctx.lineTo(nextPipe.x + GAME_CONSTANTS.PIPE_WIDTH + 10, triggerY);
      ctx.stroke();

      ctx.strokeStyle = '#0F172A';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(targetX, targetY);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.fillStyle = '#F97316';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(targetX, targetY, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // 7. Authentic Flappy Bird Sprite
    const bx = GAME_CONSTANTS.BIRD_X;
    const by = engine.birdY;
    const r = GAME_CONSTANTS.BIRD_RADIUS;
    const pitch = Math.max(-0.48, Math.min(0.85, engine.birdVel * 0.065));

    ctx.save();
    ctx.translate(bx, by);
    ctx.rotate(pitch);
    ctx.scale(1.18, 1.18);

    if (engine.lastAction === 1) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(0, 0, r + 4.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.fillStyle = agentColor;
    ctx.beginPath();
    ctx.ellipse(0, 0, r + 2, r - 1, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = 'rgba(217, 72, 15, 0.35)';
    ctx.beginPath();
    ctx.ellipse(0, 4, r, r - 6, 0, 0, Math.PI);
    ctx.fill();

    ctx.lineWidth = 2.2;
    ctx.strokeStyle = '#1E293B';
    ctx.beginPath();
    ctx.ellipse(0, 0, r + 2, r - 1, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Eye
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(6, -4, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#0F172A';
    ctx.beginPath();
    ctx.arc(7.5, -4, 2, 0, Math.PI * 2);
    ctx.fill();

    // Animated Wing
    const wingCycle = engine.lastAction === 1 ? -4 : engine.steps % 6 < 3 ? 0 : 3;
    ctx.fillStyle = '#FFFDF0';
    ctx.beginPath();
    ctx.ellipse(-4, wingCycle, 6.5, 4.2, -0.15, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Beak
    ctx.fillStyle = '#F97316';
    drawRoundedRect(ctx, 8, -1, 11, 4.5, 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#EA580C';
    drawRoundedRect(ctx, 8, 3.5, 9.5, 4, 2);
    ctx.fill();
    ctx.stroke();

    ctx.restore();
    ctx.restore(); // End world camera translate

    // 8. Authentic Scrolling Grass Bar & Ground Inside Canvas
    const grassH = 16;
    ctx.fillStyle = '#73BF2E';
    ctx.fillRect(0, G_Y, simW, grassH);

    ctx.fillStyle = '#558022';
    const scrollOffset = (engine.steps * GAME_CONSTANTS.PIPE_SPEED) % 24;
    for (let x = -24; x < simW + 24; x += 24) {
      ctx.beginPath();
      ctx.moveTo(x - scrollOffset, G_Y);
      ctx.lineTo(x - scrollOffset + 12, G_Y);
      ctx.lineTo(x - scrollOffset + 4, G_Y + grassH);
      ctx.lineTo(x - scrollOffset - 8, G_Y + grassH);
      ctx.closePath();
      ctx.fill();
    }

    ctx.strokeStyle = '#26380F';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(0, G_Y);
    ctx.lineTo(simW, G_Y);
    ctx.moveTo(0, G_Y + grassH);
    ctx.lineTo(simW + 2, G_Y + grassH);
    ctx.stroke();

    ctx.fillStyle = '#DED895';
    ctx.fillRect(0, G_Y + grassH, simW, simH - (G_Y + grassH));
    ctx.fillStyle = '#D2C975';
    for (let x = -24; x < simW + 24; x += 36) {
      ctx.fillRect(x - scrollOffset, G_Y + grassH + 10, 14, 4);
      ctx.fillRect(x - scrollOffset + 16, G_Y + grassH + 26, 10, 4);
    }

    // 9. Pixel Font In-Game Score Counter (Press Start 2P)
    ctx.save();
    ctx.textAlign = 'center';
    ctx.font = compact
      ? '400 24px "Press Start 2P", monospace'
      : '400 32px "Press Start 2P", monospace';
    ctx.lineWidth = 6;
    ctx.strokeStyle = '#0F172A';
    ctx.strokeText(`${engine.score}`, simW / 2, compact ? 44 : 52);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(`${engine.score}`, simW / 2, compact ? 44 : 52);
    ctx.restore();

    // Compact-only label for multi-agent Arena cards
    if (compact && agentName) {
      ctx.save();
      ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
      drawRoundedRect(ctx, 12, 12, 130, 26, 6);
      ctx.fill();
      ctx.fillStyle = '#F8FAFC';
      ctx.font = '600 11px "Inter", sans-serif';
      ctx.fillText(agentName, 20, 29);
      ctx.restore();
    }

    // 10. Episode-End Overlay
    if (engine.done) {
      ctx.save();
      ctx.fillStyle = 'rgba(15, 23, 42, 0.52)';
      ctx.fillRect(0, 0, simW, G_Y);

      const cardW = compact ? 240 : 320;
      const cardH = 128;
      const cardX = (simW - cardW) / 2;
      const cardY = G_Y / 2 - cardH / 2;

      ctx.fillStyle = '#FFFFFF';
      drawRoundedRect(ctx, cardX, cardY, cardW, cardH, 12);
      ctx.fill();
      ctx.strokeStyle = '#CBD5E1';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.textAlign = 'center';
      ctx.font = '400 13px "Press Start 2P", monospace';
      ctx.fillStyle = '#EA580C';
      ctx.fillText('EPISODE ENDED', simW / 2, cardY + 34);

      ctx.font = '600 13px "Inter", sans-serif';
      ctx.fillStyle = '#0F172A';
      ctx.fillText(`Pipes cleared: ${engine.score}`, simW / 2, cardY + 64);

      ctx.font = '500 12px "JetBrains Mono", monospace';
      ctx.fillStyle = '#475569';
      ctx.fillText(
        `Reward: ${engine.cumulativeReward.toFixed(1)}  •  Steps: ${engine.steps}`,
        simW / 2,
        cardY + 88
      );

      ctx.font = '600 11px "Inter", sans-serif';
      ctx.fillStyle = '#F97316';
      ctx.fillText('Click canvas or press Space to restart', simW / 2, cardY + 112);
      ctx.restore();
    }

    ctx.restore();
  }, [
    engine,
    agentName,
    agentColor,
    showExplainability,
    showTrail,
    ghostTrail,
    envTheme,
    aspectRatio,
    compact,
    logicalW,
    logicalH,
    frameTick,
  ]);

  const handleClick = () => {
    if (engine.done && onRestart) {
      onRestart();
      return;
    }
    if (onCanvasClick) {
      onCanvasClick();
    }
  };

  return (
    <div
      ref={containerRef}
      className={`relative w-full overflow-hidden rounded-xl border border-[var(--border-subtle)] bg-[#0F172A] ${
        fillBox
          ? 'aspect-[3/4] md:aspect-video md:max-h-[42vh]'
          : compact
          ? 'aspect-[3/4] max-w-[270px] mx-auto'
          : aspectRatio === '16:9'
          ? 'aspect-video w-full'
          : 'aspect-[3/4] max-w-[440px] mx-auto'
      }`}
    >
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        className={`w-full h-full block select-none ${
          onCanvasClick || (engine.done && onRestart) ? 'cursor-pointer' : ''
        }`}
      />
      {children}
    </div>
  );
};
