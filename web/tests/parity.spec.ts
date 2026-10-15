import { describe, expect, it } from 'vitest';

import type { Agent } from '../src/domain/agent';
import { createFindBallEnv, type FindBallEnv } from '../src/domain/env';
import { QLearning } from '../src/domain/qLearning';
import { LCG_PARAMS, lcgSource } from '../src/domain/rng';
import { Sarsa } from '../src/domain/sarsa';
import { Trainer } from '../src/domain/trainer';
import golden from './golden/parity.json';

/**
 * 逐位对拍：用 tools/gen_golden.py 生成的黄金轨迹，
 * 检查 TS 复刻版的每一步动作/收益/状态转移、每一集的收尾方式、以及最终 Q 表
 * 都与原 Python 算法完全一致。
 *
 * 黄金数据由「未修改的」algorithm/q_learning.py 与 algorithm/sarsa.py 产生，
 * 只把 random 换成同一串 LCG 数值、把动作集合改成有序列表（原因见生成器文件头）。
 */

const tuple = (pair: number[]): [number, number] => [pair[0], pair[1]];

function makeEnv(): FindBallEnv {
  const config = golden.config;
  return createFindBallEnv({
    width: config.width,
    height: config.height,
    start: tuple(config.start),
    goal: tuple(config.goal),
    traps: config.traps.map(tuple),
  });
}

interface ObservedStep {
  s: [number, number];
  a: string;
  r: number;
  s2: [number, number];
}

type Kind = 'qlearning' | 'sarsa';

function makeAgent(kind: Kind, rng: { next(): number }): Agent {
  const config = golden.config;
  return kind === 'qlearning'
    ? new QLearning(config.alpha, config.gamma, config.epsilon, rng)
    : new Sarsa(config.alpha, config.gamma, config.epsilon, rng);
}

interface Simulation {
  agent: Agent;
  observed: ObservedStep[];
  episodeEnds: Array<{ steps: number; end: string }>;
  draws: number;
}

/** 从种子重新开始跑指定的集数，记录每一步与随机数消耗 */
function simulate(kind: Kind, episodes: number, keepTrace: boolean): Simulation {
  const env = makeEnv();
  const base = lcgSource(golden.lcg.seed);
  let draws = 0;
  const rng = {
    next: () => {
      draws += 1;
      return base.next();
    },
  };

  const agent = makeAgent(kind, rng);
  const trainer = new Trainer(env, agent);
  trainer.start();

  const observed: ObservedStep[] = [];
  const episodeEnds: Array<{ steps: number; end: string }> = [];
  let stepsAtPreviousEpisodeEnd = 0;
  let guard = 0;

  while (trainer.episode < episodes) {
    if (++guard > 1_000_000) throw new Error('tick 次数异常，可能陷入死循环');
    const stateBeforeTick = trainer.state;
    const tick = trainer.tick();

    if (tick.type === 'step') {
      if (!keepTrace) continue;
      const transition = tick.transition;
      if (!transition) throw new Error('type=step 却没有 transition');
      observed.push({
        s: [env.x(transition.state), env.y(transition.state)],
        a: transition.action,
        r: transition.reward,
        s2: [env.x(transition.next), env.y(transition.next)],
      });
    } else {
      episodeEnds.push({
        steps: tick.steps - stepsAtPreviousEpisodeEnd,
        end: stateBeforeTick === env.goal ? 'goal' : 'trap',
      });
      stepsAtPreviousEpisodeEnd = tick.steps;
    }
  }

  return { agent, observed, episodeEnds, draws };
}

function tableToObject(agent: Agent): Record<string, Record<string, number>> {
  const table: Record<string, Record<string, number>> = {};
  for (const [state, row] of agent.qtable) {
    const rowObject: Record<string, number> = {};
    for (const [action, value] of row) rowObject[action] = value;
    table[String(state)] = rowObject;
  }
  return table;
}

describe('与 Python 原实现逐位对拍', () => {
  it('LCG 参数与 tools/gen_golden.py 完全一致', () => {
    expect({ a: golden.lcg.a, c: golden.lcg.c, mod: golden.lcg.mod }).toStrictEqual(LCG_PARAMS);
  });

  it('黄金数据本身是有内容的（覆盖了终点与陷阱两种收尾）', () => {
    const ends = golden.runs.qlearning.episodes.map((item) => item.end);
    expect(ends).toContain('goal');
    expect(ends).toContain('trap');
    expect(golden.runs.qlearning.steps.length).toBeGreaterThan(100);
    // 长跑必须真的触发过探索分支，否则这条对拍形同虚设
    expect(golden.runs.qlearning.longRun.draws).toBeGreaterThan(golden.runs.qlearning.longRun.steps);
    expect(golden.runs.sarsa.longRun.draws).toBeGreaterThan(golden.runs.sarsa.longRun.steps + 3000);
  });

  for (const kind of ['qlearning', 'sarsa'] as const) {
    it(`${kind}：30 集轨迹逐步一致，最终 Q 表逐位相同`, () => {
      const config = golden.config;
      const { agent, observed, episodeEnds, draws } = simulate(kind, config.episodes, true);

      // 1) 每一步：状态、动作、收益、下一个状态都相同（浮点数逐位比较）
      expect(observed).toStrictEqual(golden.runs[kind].steps);
      // 2) 每一集的步数与收尾原因（终点 / 陷阱）相同
      expect(episodeEnds).toStrictEqual(golden.runs[kind].episodes);
      // 3) 消耗的随机数个数相同 —— 一旦两边"探索/利用"的节奏错位，这里立刻会炸
      expect(draws).toBe(golden.runs[kind].draws);
      // 4) 最终 Q 表逐位相同
      expect(tableToObject(agent)).toStrictEqual(golden.runs[kind].finalQTable);
    });

    it(`${kind}：3000 集长跑（含探索分支）步数、随机数消耗、最终 Q 表一致`, () => {
      const longRun = golden.runs[kind].longRun;
      const { agent, draws } = simulate(kind, longRun.episodes, false);

      expect(draws).toBe(longRun.draws);
      expect(tableToObject(agent)).toStrictEqual(longRun.finalQTable);
    });
  }
});
