/**
 * Portable 32-bit Mulberry32 PRNG matching `environment/flappy_bird.py` bit-for-bit.
 * Ensures that any seed produces the exact same pipe sequence in Python and TypeScript.
 */
export class Mulberry32 {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed >>> 0;
  }

  nextFloat(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) ^ t;
    t = (t ^ (t >>> 14)) >>> 0;
    return t / 4294967296.0;
  }

  uniform(low: number, high: number): number {
    return low + (high - low) * this.nextFloat();
  }
}
