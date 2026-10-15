import type { Action } from './actions';
import { argmax, maxValue, randomKey, type Agent, type QTable, type StepResult } from './agent';
import type { GameEnv, State } from './game';
import { mathRandomSource, type RandomSource } from './rng';

/**
 * Q-Learning —— 与 algorithm/q_learning.py 逐行对应。
 *
 *   Initialize Q arbitrarily
 *   Repeat (for each episode):
 *       Initialize s
 *       Repeat (for each step of episode):
 *           Choose a from s using policy derived from Q(ε-greedy)
 *           Take action a, observe r, s'
 *           Q(s,a) = (1 - α) * Q(s,a) + α * [r + γ * maxQ(s')]
 *           s = s'
 *       until s is terminal
 *
 * 与原实现的三处对应关系：
 *   - initQTable    -> initQTable
 *   - epsilon_greedy-> epsilonGreedy（随机数从注入的 RandomSource 取）
 *   - updateQTable  -> updateQTable（保留 (1-α)·Q + α·target 的写法，勿做代数化简，
 *                      浮点结果必须与 Python 逐位一致）
 */
export class QLearning implements Agent {
  readonly kind = 'qlearning' as const;
  readonly qtable: QTable = new Map();

  constructor(
    readonly alpha: number,
    readonly gamma: number,
    readonly epsilon: number,
    private readonly rng: RandomSource = mathRandomSource,
  ) {}

  /** state 没出现过就按动作集合初始化成全 0；已存在则原样保留 */
  initQTable(state: State, actionSet: readonly Action[]): void {
    if (this.qtable.has(state)) return;
    const row = new Map<Action, number>();
    for (const action of actionSet) row.set(action, 0);
    this.qtable.set(state, row);
  }

  /** ε-greedy：rate < ε 时在合法动作里等概率随机，否则取当前最优 */
  epsilonGreedy(state: State): Action {
    const rate = this.rng.next();
    const row = this.rowOf(state);
    return rate < this.epsilon ? randomKey(row, this.rng) : argmax(row);
  }

  /** Q(s,a) <- (1-α)·Q(s,a) + α·[r + γ·maxQ(s')] */
  updateQTable(state: State, action: Action, newState: State, reward: number): number {
    const maxNext = maxValue(this.rowOf(newState));
    const row = this.rowOf(state);
    const before = row.get(action);
    if (before === undefined) throw new Error(`updateQTable: 状态 ${state} 没有动作 ${action}`);
    const after = (1 - this.alpha) * before + this.alpha * (reward + this.gamma * maxNext);
    row.set(action, after);
    return after;
  }

  /** 对应 qlStepRun 里调用的 rlObj.stepRun(...) */
  step(state: State, env: GameEnv): StepResult {
    this.initQTable(state, env.actionSet(state));

    const action = this.epsilonGreedy(state);
    const next = env.next(state, action);

    // 必须先给 s' 建表：updateQTable 会读 maxQ(s')，s' 没见过时按 0 处理
    this.initQTable(next, env.actionSet(next));

    const reward = env.reward(state, action);
    const qBefore = this.rowOf(state).get(action) as number;
    const tdTarget = reward + this.gamma * maxValue(this.rowOf(next));
    const qAfter = this.updateQTable(state, action, next, reward);

    return { state, action, reward, next, qBefore, qAfter, tdTarget };
  }

  /** qlStart 只重置球的位置，Q 表继续留着，所以这里是空操作 */
  reset(_state: State, _env: GameEnv): void {}

  greedyAction(state: State): Action {
    return argmax(this.rowOf(state));
  }

  private rowOf(state: State): Map<Action, number> {
    const row = this.qtable.get(state);
    if (row === undefined) throw new Error(`Q 表里还没有状态 ${state}（应先调用 initQTable）`);
    return row;
  }
}
