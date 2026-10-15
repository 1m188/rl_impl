import type { Action } from './actions';
import { argmax, randomKey, type Agent, type QTable, type StepResult } from './agent';
import type { GameEnv, State } from './game';
import { mathRandomSource, type RandomSource } from './rng';

/**
 * Sarsa —— 与 algorithm/sarsa.py 逐行对应。
 *
 *   Initialize Q arbitrarily
 *   Repeat (for each episode):
 *       Initialize s
 *       Choose a from s using policy derived from Q(ε-greedy)
 *       Repeat (for each step of episode):
 *           Take action a, observe r, s'
 *           Choose a' from s' using policy derived from Q(ε-greedy)
 *           Q(s,a) = (1 - α) * Q(s,a) + α * [r + γ * Q(s',a')]
 *           s = s' ; a = a'
 *       until s is terminal
 *
 * 与 Q-Learning 的唯一区别就在更新目标：这里用 Q(s',a')（下一步真的会执行的动作），
 * 而不是 maxQ(s')。所以 Sarsa 是 on-policy，Q-Learning 是 off-policy。
 */
export class Sarsa implements Agent {
  readonly kind = 'sarsa' as const;
  readonly qtable: QTable = new Map();

  /** 下一步真的要执行的动作（原实现里的 self.action） */
  private action: Action | null = null;

  constructor(
    readonly alpha: number,
    readonly gamma: number,
    readonly epsilon: number,
    private readonly rng: RandomSource = mathRandomSource,
  ) {}

  initQTable(state: State, actionSet: readonly Action[]): void {
    if (this.qtable.has(state)) return;
    const row = new Map<Action, number>();
    for (const action of actionSet) row.set(action, 0);
    this.qtable.set(state, row);
  }

  epsilonGreedy(state: State): Action {
    const rate = this.rng.next();
    const row = this.rowOf(state);
    return rate < this.epsilon ? randomKey(row, this.rng) : argmax(row);
  }

  /** 对应 sarsaStart 里的 rlObj.initAction(agentPos, getActionSet) */
  initAction(state: State, env: GameEnv): void {
    this.initQTable(state, env.actionSet(state));
    this.action = this.epsilonGreedy(state);
  }

  /** 对应 Agent 接口：Sarsa 的一集起点需要挑出第一个动作 */
  reset(state: State, env: GameEnv): void {
    this.initAction(state, env);
  }

  /** Q(s,a) <- (1-α)·Q(s,a) + α·[r + γ·Q(s',a')] */
  updateQTable(
    state: State,
    action: Action,
    newState: State,
    newAction: Action,
    reward: number,
  ): number {
    const nextRow = this.rowOf(newState);
    const nextValue = nextRow.get(newAction);
    if (nextValue === undefined) {
      throw new Error(`updateQTable: 状态 ${newState} 没有动作 ${newAction}`);
    }

    const row = this.rowOf(state);
    const before = row.get(action);
    if (before === undefined) throw new Error(`updateQTable: 状态 ${state} 没有动作 ${action}`);
    const after = (1 - this.alpha) * before + this.alpha * (reward + this.gamma * nextValue);
    row.set(action, after);
    return after;
  }

  /** 对应 sarsaStepRun 里调用的 rlObj.stepRun(...) */
  step(state: State, env: GameEnv): StepResult {
    if (this.action === null) {
      throw new Error('Sarsa: 还没有初始化动作（一集开始时必须先调用 initAction/reset）');
    }

    this.initQTable(state, env.actionSet(state));

    // 这里的动作是上一步（或 initAction）挑好的，本步必定执行
    const action = this.action;
    const next = env.next(state, action);
    const reward = env.reward(state, action);

    this.initQTable(next, env.actionSet(next));

    // 为 s' 挑下一个动作 a'，并用 Q(s',a') 作为更新目标
    const newAction = this.epsilonGreedy(next);

    const qBefore = this.rowOf(state).get(action) as number;
    const tdTarget = reward + this.gamma * (this.rowOf(next).get(newAction) as number);
    const qAfter = this.updateQTable(state, action, next, newAction, reward);

    this.action = newAction;

    return { state, action, reward, next, qBefore, qAfter, tdTarget };
  }

  greedyAction(state: State): Action {
    return argmax(this.rowOf(state));
  }

  private rowOf(state: State): Map<Action, number> {
    const row = this.qtable.get(state);
    if (row === undefined) throw new Error(`Q 表里还没有状态 ${state}（应先调用 initQTable）`);
    return row;
  }
}
