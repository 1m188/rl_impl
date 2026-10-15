import { ACTION_ORDER, DELTA, type Action } from './actions';
import type { GameEnv, State } from './game';

export type { GameEnv, State } from './game';

/**
 * 找红球的游戏规则 —— 对应 findball.py 注入给算法的那几个回调：
 *   getActionSet / getNewState / getReward / updateState(由 Trainer 负责)
 * 这里把它们收在一个对象里，逻辑逐行对齐原实现。
 */
export interface FindBallEnv extends GameEnv {
  readonly width: number;
  readonly height: number;
  readonly start: State;
  readonly goal: State;
  readonly traps: ReadonlySet<State>;

  pack(x: number, y: number): State;
  x(state: State): number;
  y(state: State): number;
}

export interface FindBallConfig {
  width?: number;
  height?: number;
  start?: readonly [number, number];
  goal?: readonly [number, number];
  traps?: ReadonlyArray<readonly [number, number]>;
}

/** findball.py 里写死的场景：4x4，(0,0) 出发，(2,2) 是终点，三个黑格陷阱 */
export const DEFAULT_FIND_BALL_CONFIG: Required<FindBallConfig> = {
  width: 4,
  height: 4,
  start: [0, 0],
  goal: [2, 2],
  traps: [
    [1, 1],
    [1, 2],
    [2, 1],
  ],
};

export function createFindBallEnv(config: FindBallConfig = {}): FindBallEnv {
  const width = config.width ?? DEFAULT_FIND_BALL_CONFIG.width;
  const height = config.height ?? DEFAULT_FIND_BALL_CONFIG.height;
  const startXY = config.start ?? DEFAULT_FIND_BALL_CONFIG.start;
  const goalXY = config.goal ?? DEFAULT_FIND_BALL_CONFIG.goal;
  const trapsXY = config.traps ?? DEFAULT_FIND_BALL_CONFIG.traps;

  const pack = (x: number, y: number): State => y * width + x;
  const x = (state: State): number => state % width;
  const y = (state: State): number => Math.floor(state / width);

  const start = pack(startXY[0], startXY[1]);
  const goal = pack(goalXY[0], goalXY[1]);
  const traps: ReadonlySet<State> = new Set(trapsXY.map(([tx, ty]) => pack(tx, ty)));

  const actionSet = (state: State): Action[] => {
    const cx = x(state);
    const cy = y(state);
    const actions: Action[] = [];
    for (const action of ACTION_ORDER) {
      const nx = cx + DELTA[action][0];
      const ny = cy + DELTA[action][1];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) actions.push(action);
    }
    return actions;
  };

  const next = (state: State, action: Action): State =>
    pack(x(state) + DELTA[action][0], y(state) + DELTA[action][1]);

  const reward = (state: State, action: Action): number => {
    const newState = next(state, action);
    if (newState === goal) return 10;
    if (traps.has(newState)) return -10;
    const dx = x(newState) - x(goal);
    const dy = y(newState) - y(goal);
    // 原式：-((dx ** 2 + dy ** 2) ** 0.5)，Math.sqrt 与之逐位一致（见 tests/parity.spec.ts）
    return -Math.sqrt(dx * dx + dy * dy);
  };

  const isTerminal = (state: State): boolean => state === goal || traps.has(state);

  /** 终止原因：终点 / 陷阱；找红球没有"超时"这种收尾 */
  const terminalReason = (state: State): string | null => {
    if (state === goal) return 'goal';
    if (traps.has(state)) return 'trap';
    return null;
  };

  const describe = (state: State): string => `(${x(state)}, ${y(state)})`;

  /** 开一局：回到起点（找红球没有随机性，所以每次都一样） */
  const reset = (): State => start;

  return {
    width,
    height,
    start,
    goal,
    traps,
    pack,
    x,
    y,
    actionSet,
    next,
    reward,
    isTerminal,
    terminalReason,
    describe,
    reset,
  };
}
