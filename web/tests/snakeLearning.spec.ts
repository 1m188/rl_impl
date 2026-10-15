import { describe, expect, it } from 'vitest';

import type { Agent } from '../src/domain/agent';
import { QLearning } from '../src/domain/qLearning';
import { lcgSource } from '../src/domain/rng';
import { Sarsa } from '../src/domain/sarsa';
import { createSnakeEnv } from '../src/domain/snake';
import { Trainer } from '../src/domain/trainer';

/**
 * 学习效果断言：确认"压缩状态 + 距离变化量塑形"这套设计真的能让
 * 表格法 Q-Learning / Sarsa 学会吃食物，而不是只能跑起来。
 *
 * 超参就是界面上贪吃蛇页签的默认值（α=0.5, γ=0.9, ε=0.1）。
 */

const EPISODES = 2000;
const MAX_STEPS = 300;
const ENV_SEED = 12345;
const AGENT_SEED = 999;

interface Outcome {
  /** 每集吃到的食物个数 */
  scores: number[];
  /** 每集活了多少步 */
  lengths: number[];
  ends: Record<string, number>;
  states: number;
}

function train(kind: 'ql' | 'sarsa'): Outcome {
  const env = createSnakeEnv();
  env.useRandom(lcgSource(ENV_SEED));
  const agent: Agent =
    kind === 'ql'
      ? new QLearning(0.5, 0.9, 0.1, lcgSource(AGENT_SEED))
      : new Sarsa(0.5, 0.9, 0.1, lcgSource(AGENT_SEED));

  const trainer = new Trainer(env, agent, { maxStepsPerEpisode: MAX_STEPS });
  trainer.start();

  const scores: number[] = [];
  const lengths: number[] = [];
  const ends: Record<string, number> = {};
  let score = 0;

  while (trainer.history.length < EPISODES) {
    const tick = trainer.tick();
    if (tick.type === 'step') score = env.snapshot().score;
    else {
      scores.push(score);
      lengths.push(trainer.history[trainer.history.length - 1].steps);
      ends[tick.endReason ?? '?'] = (ends[tick.endReason ?? '?'] ?? 0) + 1;
      score = 0;
    }
  }

  return { scores, lengths, ends, states: agent.qtable.size };
}

const avg = (values: number[]) =>
  values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;

describe.each([
  ['Q-Learning', 'ql'],
  ['Sarsa', 'sarsa'],
] as const)('贪吃蛇的学习效果：%s', (_label, kind) => {
  const outcome = train(kind);

  it(`${EPISODES} 集之后明显比刚开始会吃（前 200 集 vs 后 200 集）`, () => {
    const early = avg(outcome.scores.slice(0, 200));
    const late = avg(outcome.scores.slice(-200));
    expect(late).toBeGreaterThan(early * 1.5);
    expect(late).toBeGreaterThan(4);
  });

  it('活得也更久（平均存活步数上升）', () => {
    const early = avg(outcome.lengths.slice(0, 200));
    const late = avg(outcome.lengths.slice(-200));
    expect(late).toBeGreaterThan(early);
    expect(late).toBeGreaterThan(20);
  });

  it('Q 表真的在用压缩状态：状态数远小于整个棋盘的局面数', () => {
    expect(outcome.states).toBeGreaterThan(300);
    expect(outcome.states).toBeLessThan(16384);
  });

  it('大部分回合是因为撞死结束，说明它在真的玩而不是被超时截断', () => {
    expect(outcome.ends.death ?? 0).toBeGreaterThan(EPISODES * 0.5);
  });
});
