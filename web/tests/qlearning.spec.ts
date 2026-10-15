import { describe, expect, it } from 'vitest';

import { createFindBallEnv } from '../src/domain/env';
import { QLearning } from '../src/domain/qLearning';
import { scriptedSource } from '../src/domain/rng';

const env = createFindBallEnv();
/** 这些用例不该消耗随机数，给一个空的来源当作"用到就报错"的哨兵 */
const noRandom = () => scriptedSource([]);

describe('QLearning', () => {
  it('动作集合按 ACTION_ORDER 过滤越界动作（起点 (0,0) 只剩 DOWN/RIGHT）', () => {
    expect(env.actionSet(env.start)).toStrictEqual(['DOWN', 'RIGHT']);
    expect(env.actionSet(env.pack(1, 1))).toStrictEqual(['UP', 'DOWN', 'LEFT', 'RIGHT']);
  });

  it('initQTable：新状态按动作集合初始化为 0，已存在则原样保留', () => {
    const agent = new QLearning(0.5, 0.5, 0.01, noRandom());
    agent.initQTable(0, ['DOWN', 'RIGHT']);
    expect([...agent.qtable.get(0)!]).toStrictEqual([
      ['DOWN', 0],
      ['RIGHT', 0],
    ]);

    agent.qtable.get(0)!.set('DOWN', 7);
    agent.initQTable(0, ['DOWN', 'RIGHT']);
    expect(agent.qtable.get(0)!.get('DOWN')).toBe(7);
  });

  it('epsilonGreedy：rate < ε 时随机取动作，否则取当前最优', () => {
    // 探索：rate=0.1 < 0.5，再取一个 0.99 -> floor(0.99*2)=1 -> 第二个动作
    const exploring = new QLearning(0.5, 0.5, 0.5, scriptedSource([0.1, 0.99]));
    exploring.initQTable(0, ['DOWN', 'RIGHT']);
    expect(exploring.epsilonGreedy(0)).toBe('RIGHT');

    // 利用：rate=0.9 >= 0.5
    const exploiting = new QLearning(0.5, 0.5, 0.5, scriptedSource([0.9]));
    exploiting.initQTable(0, ['DOWN', 'RIGHT']);
    exploiting.qtable.get(0)!.set('DOWN', 3);
    expect(exploiting.epsilonGreedy(0)).toBe('DOWN');
  });

  it('并列时取动作顺序靠前的那个（对应 Python 的 max(dict, key=dict.get)）', () => {
    const agent = new QLearning(0.5, 0.5, 0.01, noRandom());
    agent.initQTable(0, ['DOWN', 'RIGHT']);
    expect(agent.greedyAction(0)).toBe('DOWN');
  });

  it("updateQTable：Q(s,a) = (1-α)Q(s,a) + α[r + γ·maxQ(s')]", () => {
    const agent = new QLearning(0.5, 0.5, 0.01, noRandom());
    agent.initQTable(0, ['DOWN', 'RIGHT']);
    agent.initQTable(1, ['DOWN', 'RIGHT']);
    agent.qtable.get(0)!.set('RIGHT', 4);
    agent.qtable.get(1)!.set('DOWN', 6);

    // (1-0.5)*4 + 0.5*(-2 + 0.5*6) = 2 + 0.5*1 = 2.5
    expect(agent.updateQTable(0, 'RIGHT', 1, -2)).toBe(2.5);
  });

  it("step：先给 s' 建表再更新（没见过的 s' 按 0 参与计算）", () => {
    const agent = new QLearning(0.5, 0.5, 0.01, scriptedSource([0.9]));
    const result = agent.step(env.start, env);

    expect(result.action).toBe('DOWN'); // 全 0 并列 -> 顺序靠前
    expect(result.next).toBe(env.pack(0, 1));
    expect(agent.qtable.has(result.next)).toBe(true);
    expect(result.qBefore).toBe(0);
    expect(result.tdTarget).toBe(result.reward); // maxQ(s') 此刻还是 0
    expect(result.qAfter).toBe(0.5 * result.reward);
  });

  it('reset 是空操作：Q 表跨集保留（对应 qlStart 只挪球）', () => {
    const agent = new QLearning(0.5, 0.5, 0.01, noRandom());
    agent.initQTable(0, ['DOWN', 'RIGHT']);
    agent.qtable.get(0)!.set('DOWN', 1.5);
    agent.reset(0, env);
    expect(agent.qtable.get(0)!.get('DOWN')).toBe(1.5);
  });
});
