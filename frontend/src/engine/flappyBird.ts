import { Mulberry32 } from './prng';

export const GAME_CONSTANTS = {
  WIDTH: 400,
  HEIGHT: 600,
  GROUND_Y: 520,
  BIRD_X: 80,
  BIRD_RADIUS: 14,
  GRAVITY: 0.45,
  FLAP_STRENGTH: -7.5,
  MAX_VEL_Y: 10.0,
  MIN_VEL_Y: -9.0,
  PIPE_WIDTH: 56,
  PIPE_GAP: 120,
  PIPE_SPEED: 3.0,
  PIPE_SPAWN_SPACING: 210,
  FIRST_PIPE_X: 320,
  GAP_MIN_Y: 80,
  GAP_MAX_Y: 310,
} as const;

export interface Pipe {
  x: number;
  gapTopY: number;
  gapBottomY: number;
  gapCenterY: number;
  passed: boolean;
}

export interface TrailPoint {
  x: number;
  y: number;
  flapped: boolean;
}

export class FlappyBirdEngine {
  public birdY: number = GAME_CONSTANTS.GROUND_Y * 0.45;
  public birdVel: number = 0;
  public pipes: Pipe[] = [];
  public score: number = 0;
  public steps: number = 0;
  public done: boolean = false;
  public cumulativeReward: number = 0;
  public lastAction: 0 | 1 = 0;
  public trail: TrailPoint[] = [];
  private prng: Mulberry32;
  public currentSeed: number;

  constructor(seed: number = 42) {
    this.currentSeed = seed;
    this.prng = new Mulberry32(seed);
    this.reset(seed);
  }

  private spawnPipe(x: number): Pipe {
    const gapTopY =
      Math.round(
        this.prng.uniform(GAME_CONSTANTS.GAP_MIN_Y, GAME_CONSTANTS.GAP_MAX_Y) * 100
      ) / 100;
    const gapBottomY = gapTopY + GAME_CONSTANTS.PIPE_GAP;
    return {
      x,
      gapTopY,
      gapBottomY,
      gapCenterY: 0.5 * (gapTopY + gapBottomY),
      passed: false,
    };
  }

  public reset(seed?: number): number[] {
    if (seed !== undefined) {
      this.currentSeed = seed;
    }
    this.prng = new Mulberry32(this.currentSeed);
    this.birdY = GAME_CONSTANTS.GROUND_Y * 0.45;
    this.birdVel = 0;
    this.score = 0;
    this.steps = 0;
    this.done = false;
    this.cumulativeReward = 0;
    this.lastAction = 0;
    this.trail = [{ x: GAME_CONSTANTS.BIRD_X, y: this.birdY, flapped: false }];

    this.pipes = [
      this.spawnPipe(GAME_CONSTANTS.FIRST_PIPE_X),
      this.spawnPipe(GAME_CONSTANTS.FIRST_PIPE_X + GAME_CONSTANTS.PIPE_SPAWN_SPACING),
      this.spawnPipe(GAME_CONSTANTS.FIRST_PIPE_X + 2 * GAME_CONSTANTS.PIPE_SPAWN_SPACING),
      this.spawnPipe(GAME_CONSTANTS.FIRST_PIPE_X + 3 * GAME_CONSTANTS.PIPE_SPAWN_SPACING),
      this.spawnPipe(GAME_CONSTANTS.FIRST_PIPE_X + 4 * GAME_CONSTANTS.PIPE_SPAWN_SPACING),
      this.spawnPipe(GAME_CONSTANTS.FIRST_PIPE_X + 5 * GAME_CONSTANTS.PIPE_SPAWN_SPACING),
      this.spawnPipe(GAME_CONSTANTS.FIRST_PIPE_X + 6 * GAME_CONSTANTS.PIPE_SPAWN_SPACING),
    ];

    return this.getState();
  }

  public getNextPipe(): Pipe {
    const birdLeft = GAME_CONSTANTS.BIRD_X - GAME_CONSTANTS.BIRD_RADIUS;
    for (const pipe of this.pipes) {
      if (pipe.x + GAME_CONSTANTS.PIPE_WIDTH >= birdLeft) {
        return pipe;
      }
    }
    return this.pipes[0];
  }

  /**
   * Returns the 5D normalized state vector identical to Python `FlappyBirdEnv.get_state()`:
   * [bird_y / GROUND_Y, bird_vel / MAX_VEL_Y, dx, dy_upper, dy_lower]
   */
  public getState(): number[] {
    const nextPipe = this.getNextPipe();
    const dx =
      (nextPipe.x + GAME_CONSTANTS.PIPE_WIDTH - GAME_CONSTANTS.BIRD_X) /
      GAME_CONSTANTS.WIDTH;
    const dyUpper = (this.birdY - nextPipe.gapTopY) / GAME_CONSTANTS.GROUND_Y;
    const dyLower = (nextPipe.gapBottomY - this.birdY) / GAME_CONSTANTS.GROUND_Y;

    return [
      this.birdY / GAME_CONSTANTS.GROUND_Y,
      this.birdVel / GAME_CONSTANTS.MAX_VEL_Y,
      dx,
      dyUpper,
      dyLower,
    ];
  }

  private checkCollision(): boolean {
    if (
      this.birdY - GAME_CONSTANTS.BIRD_RADIUS <= 0 ||
      this.birdY + GAME_CONSTANTS.BIRD_RADIUS >= GAME_CONSTANTS.GROUND_Y
    ) {
      return true;
    }

    const bx = GAME_CONSTANTS.BIRD_X;
    const by = this.birdY;
    const r = GAME_CONSTANTS.BIRD_RADIUS - 2.0;

    for (const pipe of this.pipes) {
      if (bx + r > pipe.x && bx - r < pipe.x + GAME_CONSTANTS.PIPE_WIDTH) {
        if (by - r < pipe.gapTopY || by + r > pipe.gapBottomY) {
          return true;
        }
      }
    }
    return false;
  }

  public step(action: 0 | 1): {
    state: number[];
    reward: number;
    done: boolean;
    passedPipe: boolean;
  } {
    if (this.done) {
      return {
        state: this.getState(),
        reward: 0,
        done: true,
        passedPipe: false,
      };
    }

    this.lastAction = action;
    if (action === 1) {
      this.birdVel = GAME_CONSTANTS.FLAP_STRENGTH;
    } else {
      this.birdVel = Math.min(
        GAME_CONSTANTS.MAX_VEL_Y,
        this.birdVel + GAME_CONSTANTS.GRAVITY
      );
    }

    this.birdY += this.birdVel;
    this.steps += 1;

    // Shift trail points left with pipe speed so trail shows spatial path
    for (const pt of this.trail) {
      pt.x -= GAME_CONSTANTS.PIPE_SPEED;
    }
    this.trail.push({
      x: GAME_CONSTANTS.BIRD_X,
      y: this.birdY,
      flapped: action === 1,
    });
    if (this.trail.length > 48) {
      this.trail.shift();
    }

    let passedPipe = false;
    for (const pipe of this.pipes) {
      pipe.x -= GAME_CONSTANTS.PIPE_SPEED;
      if (!pipe.passed && pipe.x + GAME_CONSTANTS.PIPE_WIDTH < GAME_CONSTANTS.BIRD_X) {
        pipe.passed = true;
        this.score += 1;
        passedPipe = true;
      }
    }

    if (this.pipes.length > 0 && this.pipes[0].x + GAME_CONSTANTS.PIPE_WIDTH < -10) {
      this.pipes.shift();
      const rightmostX = Math.max(...this.pipes.map((p) => p.x));
      this.pipes.push(this.spawnPipe(rightmostX + GAME_CONSTANTS.PIPE_SPAWN_SPACING));
    }

    const collision = this.checkCollision();
    this.done = collision;

    let reward = 0.1;
    if (collision) {
      reward = -10.0;
    } else if (passedPipe) {
      reward += 1.0;
    }

    this.cumulativeReward = Math.round((this.cumulativeReward + reward) * 100) / 100;

    return {
      state: this.getState(),
      reward,
      done: this.done,
      passedPipe,
    };
  }
}
