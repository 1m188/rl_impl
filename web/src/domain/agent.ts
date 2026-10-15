import type { Action } from './actions';
import type { FindBallEnv, State } from './env';

/** Q 表：state -> { action -> 价值 }。原实现是 dict[state][action]。 */
export type QTable = Map<State, Map<Action, number>>;

/** 一次 stepRun 的完整结果，便于测试断言与阶段二把更新过程画出来 */
export interface StepResult {
  /** 更新前的状态 s */
  state: State;
  /** 本步实际选择的动作 a */
  action: Action;
  /** 本步获得的即时收益 r */
  reward: number;
  /** 转移后的状态 s' */
  next: State;
  /** 更新前的 Q(s,a) */
  qBefore: number;
  /** 更新后的 Q(s,a) */
  qAfter: number;
  /** 更新用的目标值：Q-Learning 是 r + γ·maxQ(s')，Sarsa 是 r + γ·Q(s',a') */
  tdTarget: number;
}

/** 让两个算法可以互换的公共接口（原实现里它们只是"长得像"，并没有共同基类） */
export interface Agent {
  readonly kind: 'qlearning' | 'sarsa';
  readonly qtable: QTable;
  /**
   * 一集开始时的准备。对应 qlStart / sarsaStart：
   * Q-Learning 无事可做；Sarsa 需要 initAction（挑出第一步要执行的动作）。
   */
  reset(state: State, env: FindBallEnv): void;
  /** 对应 stepRun：更新 Q 表并把"下一个状态"作为结果返回（状态的推进由 Trainer 完成） */
  step(state: State, env: FindBallEnv): StepResult;
  /** 不含探索的策略（取 argmax）。原实现没有这个方法，是为可视化与测试新增的读取入口。 */
  greedyAction(state: State): Action;
}

/** 对应 max(ar, key=ar.get)：并列时取先插入的那个（= 动作顺序里靠前的） */
export function argmax(row: Map<Action, number>): Action {
  let bestAction: Action | undefined;
  let bestValue = -Infinity;
  for (const [action, value] of row) {
    if (bestAction === undefined || value > bestValue) {
      bestAction = action;
      bestValue = value;
    }
  }
  if (bestAction === undefined) throw new Error('argmax: 该状态的 Q 表为空');
  return bestAction;
}

/** 对应 max(dict.values()) */
export function maxValue(row: Map<Action, number>): number {
  let best = -Infinity;
  for (const value of row.values()) {
    if (value > best) best = value;
  }
  if (best === -Infinity) throw new Error('maxValue: 该状态的 Q 表为空');
  return best;
}

/** 对应 random.sample(ar.keys(), 1)[0]：在合法动作里等概率取一个 */
export function randomKey(row: Map<Action, number>, rng: { next(): number }): Action {
  const keys = [...row.keys()];
  const index = Math.floor(rng.next() * keys.length);
  return keys[index];
}
