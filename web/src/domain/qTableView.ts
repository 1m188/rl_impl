import type { Agent } from './agent';
import { ACTION_ORDER, type Action } from './actions';
import type { FindBallEnv, State } from './env';

/**
 * Q 表的只读视图：把 Map 形状的 Q 表摊平成界面好用的结构。
 *
 * 放在 domain 里而不是组件里，是为了让它能被单独测试，也让组件只负责画。
 * 这里只读，不会改动算法任何状态。
 */

export interface StateView {
  state: State;
  x: number;
  y: number;
  /** 终点或陷阱：Q 表里可能有行，但值永远是 0（一集到这儿就结束了），画出来会误导 */
  terminal: boolean;
  /** Q 表里是否已经有这个状态 */
  known: boolean;
  values: Partial<Record<Action, number>>;
  /** max Q(s,·)，热力图用的就是这个 */
  best: number | null;
  worst: number | null;
  /** argmax，并列时取动作顺序靠前的那个（与算法里的 greedyAction 规则一致） */
  policy: Action | null;
}

export interface QTableView {
  cells: StateView[];
  /** 所有「已知状态」的 best 的取值范围，用于热力归一化 */
  min: number;
  max: number;
  knownCount: number;
  stateCount: number;
}

/** 棋盘叠加层的显示方式 */
export type DisplayMode = 'none' | 'policy' | 'heat' | 'both';

export const DISPLAY_MODES: ReadonlyArray<{ value: DisplayMode; label: string }> = [
  { value: 'both', label: '箭头 + 热力' },
  { value: 'policy', label: '只看策略箭头' },
  { value: 'heat', label: '只看热力图' },
  { value: 'none', label: '不显示' },
];

export function showsPolicy(mode: DisplayMode): boolean {
  return mode === 'policy' || mode === 'both';
}

export function showsHeat(mode: DisplayMode): boolean {
  return mode === 'heat' || mode === 'both';
}

export function readQTableView(agent: Agent | null, env: FindBallEnv): QTableView {
  const cells: StateView[] = [];
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let knownCount = 0;

  for (let y = 0; y < env.height; y += 1) {
    for (let x = 0; x < env.width; x += 1) {
      const state = env.pack(x, y);
      const row = agent ? agent.qtable.get(state) : undefined;

      const values: Partial<Record<Action, number>> = {};
      let best: number | null = null;
      let worst: number | null = null;
      let policy: Action | null = null;

      if (row) {
        for (const action of ACTION_ORDER) {
          const value = row.get(action);
          if (value === undefined) continue;
          values[action] = value;
          if (best === null || value > best) {
            best = value;
            policy = action;
          }
          if (worst === null || value < worst) worst = value;
        }
      }

      if (best !== null) {
        knownCount += 1;
        if (best < min) min = best;
        if (best > max) max = best;
      }

      cells.push({
        state,
        x,
        y,
        terminal: env.isTerminal(state),
        known: row !== undefined,
        values,
        best,
        worst,
        policy,
      });
    }
  }

  return {
    cells,
    min: knownCount === 0 ? 0 : min,
    max: knownCount === 0 ? 0 : max,
    knownCount,
    stateCount: env.width * env.height,
  };
}

/** 把 best 归一化到 0..1（供热力图上色）。所有值一样时返回 0.5，避免除零。 */
export function normalizeHeat(value: number, min: number, max: number): number {
  if (!(max > min)) return 0.5;
  const ratio = (value - min) / (max - min);
  return Math.min(1, Math.max(0, ratio));
}

/** Q 表里已知状态的 best 平均值，用来显示"学到什么程度了" */
export function averageBest(view: QTableView): number | null {
  const known = view.cells.filter((cell) => cell.best !== null);
  if (known.length === 0) return null;
  const sum = known.reduce((acc, cell) => acc + (cell.best as number), 0);
  return sum / known.length;
}
