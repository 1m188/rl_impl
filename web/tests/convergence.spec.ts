import { describe, expect, it } from 'vitest';

import type { Agent } from '../src/domain/agent';
import { createFindBallEnv, type FindBallEnv, type State } from '../src/domain/env';
import { QLearning } from '../src/domain/qLearning';
import { lcgSource } from '../src/domain/rng';
import { Sarsa } from '../src/domain/sarsa';
import { Trainer } from '../src/domain/trainer';

/**
 * 收敛性断言：这些性质与我用原 Python 代码跑出来的结论一致
 * （见 web/README.md 里的实测记录），是"算法真的学会了"而不是"能跑"的证据。
 */

const SEEDS = [1, 2, 3, 5, 8, 13, 21, 34];
const TRAIN_EPISODES = 30;
const OPTIMAL_SAFE_STEPS = 6; // (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(2,2) 之类的安全最短路

function makeAgent(kind: 'qlearning' | 'sarsa', seed: number): Agent {
  const rng = lcgSource(seed);
  return kind === 'qlearning'
    ? new QLearning(0.5, 0.5, 0.01, rng)
    : new Sarsa(0.5, 0.5, 0.01, rng);
}

/** 用训练好的 Q 表走纯贪心路线（不再探索），看看它到底会不会玩 */
function greedyRollout(env: FindBallEnv, agent: Agent): { path: State[]; end: 'goal' | 'trap' | 'loop' } {
  const path: State[] = [env.start];
  let state = env.start;
  for (let i = 0; i < 50; i += 1) {
    if (state === env.goal) return { path, end: 'goal' };
    if (env.traps.has(state)) return { path, end: 'trap' };
    state = env.next(state, agent.greedyAction(state));
    path.push(state);
  }
  return { path, end: 'loop' };
}

describe.each(['qlearning', 'sarsa'] as const)('%s 的收敛性', (kind) => {
  it(`${TRAIN_EPISODES} 集后，贪心策略能走安全最短路到达终点（%d 个种子）`, () => {
    for (const seed of SEEDS) {
      const env = createFindBallEnv();
      const agent = makeAgent(kind, seed);
      const trainer = new Trainer(env, agent);
      trainer.start();
      while (trainer.episode < TRAIN_EPISODES) trainer.tick();

      const { path, end } = greedyRollout(env, agent);
      const pretty = path.map((s) => `(${env.x(s)},${env.y(s)})`).join('->');
      const reason = `seed=${seed} 结束于 ${end}，路径 ${pretty}`;

      expect(end, reason).toBe('goal');
      expect(path.length - 1, reason).toBe(OPTIMAL_SAFE_STEPS);
      // 路上绝不会踩进陷阱：-10 的惩罚远远压过距离塑形（-1 ~ -2.83）
      expect(path.some((s) => env.traps.has(s)), reason).toBe(false);
    }
  });

  it('整个状态空间只有 16 个格子 —— 这是它能学会的根本原因', () => {
    const env = createFindBallEnv();
    const agent = makeAgent(kind, 7);
    const trainer = new Trainer(env, agent);
    trainer.start();
    while (trainer.episode < TRAIN_EPISODES) trainer.tick();

    expect(env.width * env.height).toBe(16);
    expect(agent.qtable.size).toBeLessThanOrEqual(16);
    // 终点与陷阱也会被建表（原实现里 updateQTable 会读到它们），但它们的值不会被更新
    for (const terminal of [env.goal, ...env.traps]) {
      if (agent.qtable.has(terminal)) {
        expect([...agent.qtable.get(terminal)!.values()].every((v) => v === 0)).toBe(true);
      }
    }
  });

  it('每集都会结束（终点或陷阱），不会卡在原地', () => {
    const env = createFindBallEnv();
    const agent = makeAgent(kind, 11);
    const trainer = new Trainer(env, agent);
    trainer.start();

    let ticks = 0;
    while (trainer.episode < 5) {
      trainer.tick();
      ticks += 1;
      expect(ticks).toBeLessThan(500);
    }
    expect(trainer.steps).toBeGreaterThan(0);
  });
});
