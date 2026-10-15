import { describe, expect, it } from 'vitest';

import { createFindBallEnv } from '../src/domain/env';
import { QLearning } from '../src/domain/qLearning';
import { scriptedSource } from '../src/domain/rng';
import { Sarsa } from '../src/domain/sarsa';

const env = createFindBallEnv();
const noRandom = () => scriptedSource([]);

describe('Sarsa', () => {
  it('initAction 决定第一步动作；没初始化就 step 会直接报错', () => {
    // 三次抽取：initAction 的 rate + 随机选动作，以及 step 末尾为 (0,1) 挑 a' 的 rate
    const agent = new Sarsa(0.5, 0.5, 0.5, scriptedSource([0.1, 0.0, 0.9]));
    expect(() => agent.step(env.start, env)).toThrow(/初始化动作/);

    agent.initAction(env.start, env); // rate=0.1 < 0.5 探索，pick 0.0 -> 第一个动作 DOWN
    const result = agent.step(env.start, env);
    expect(result.action).toBe('DOWN');
    expect(result.next).toBe(env.pack(0, 1));
  });

  it('step 执行的是上一步为 s\' 挑好的动作（on-policy：说到做到）', () => {
    // initAction: rate=0.1 探索 + pick 0.0 -> (0,0) 的动作表 [DOWN,RIGHT] -> DOWN
    // step1: 为 (0,1) 挑 a'：rate=0.1 探索 + pick 0.0 -> (0,1) 的动作表 [UP,DOWN,RIGHT] -> UP
    // step2: 为 (0,0) 挑 a'：rate=0.9 利用（不再多消耗随机数）
    const agent = new Sarsa(0.5, 0.5, 0.5, scriptedSource([0.1, 0.0, 0.1, 0.0, 0.9]));
    agent.initAction(env.start, env);

    const first = agent.step(env.start, env);
    expect(first.action).toBe('DOWN');

    const second = agent.step(first.next, env);
    expect(second.action).toBe('UP'); // 上一步就替这个状态挑好了，本步必定执行
    expect(second.next).toBe(env.pack(0, 0));
  });

  it("updateQTable 用的是 Q(s',a')，而不是 maxQ(s') —— 这就是 on/off-policy 的分水岭", () => {
    const sarsa = new Sarsa(0.5, 0.5, 0.01, noRandom());
    const qlearning = new QLearning(0.5, 0.5, 0.01, noRandom());
    for (const agent of [sarsa, qlearning]) {
      agent.initQTable(0, ['DOWN', 'RIGHT']);
      agent.initQTable(1, ['DOWN', 'RIGHT']);
      agent.qtable.get(0)!.set('RIGHT', 4);
      agent.qtable.get(1)!.set('DOWN', 6);
      agent.qtable.get(1)!.set('RIGHT', 1);
    }

    // Sarsa 用 Q(s',a'=RIGHT)=1：(1-0.5)*4 + 0.5*(-2 + 0.5*1) = 2 - 0.75 = 1.25
    expect(sarsa.updateQTable(0, 'RIGHT', 1, 'RIGHT', -2)).toBe(1.25);
    // Q-Learning 用 maxQ(s')=6：(1-0.5)*4 + 0.5*(-2 + 0.5*6) = 2 + 0.5 = 2.5
    expect(qlearning.updateQTable(0, 'RIGHT', 1, -2)).toBe(2.5);
  });

  it('reset 会重新挑动作（对应 sarsaStart 里的 initAction）', () => {
    const agent = new Sarsa(0.5, 0.5, 0.5, scriptedSource([0.1, 0.99, 0.9]));
    agent.reset(env.start, env); // rate=0.1 探索，pick 0.99 -> 第二个动作 RIGHT
    const result = agent.step(env.start, env);
    expect(result.action).toBe('RIGHT');
  });
});
