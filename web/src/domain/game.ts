import type { Action } from './actions';
import type { RandomSource } from './rng';

/**
 * 状态的类型。
 *
 * 算法只把它当"不透明的 Map 键"使用，从不做算术，所以每个游戏可以自己决定怎么编码：
 *   - 找红球：格子编号 y * width + x（棋盘内容是固定的，格子编号就够描述局面）
 *   - 贪吃蛇：把蛇头局部局势压缩成一个特征键（见 snake.ts）
 */
export type State = number;

/**
 * 游戏环境 —— 算法只需要这四件事：合法动作、状态转移、即时收益、终止判定。
 * 两个游戏共用同一份 QLearning / Sarsa / Trainer，靠的就是这个接口。
 */
export interface GameEnv {
  /**
   * 开一局：重置局面并返回初始状态。
   *
   * 找红球是确定性的（永远从左上角出发），贪吃蛇则要重新随机撒食物，
   * 所以这里必须是方法而不是一个常量。
   */
  reset(): State;

  /** 该状态下可以选择的动作 */
  actionSet(state: State): Action[];

  /**
   * 执行动作后的下一个状态。
   *
   * 找红球是纯函数；贪吃蛇的局面（蛇身、食物）保存在环境内部，
   * 传进来的 state 只是"当前局面的特征键"，转移由内部局面推进完成。
   */
  next(state: State, action: Action): State;

  /** 上一步 (state, action) 得到的即时收益 */
  reward(state: State, action: Action): number;

  isTerminal(state: State): boolean;

  /** 终止原因，用于统计与提示；未终止返回 null */
  terminalReason(state: State): string | null;

  /** 人类可读的状态描述（单步细节面板用） */
  describe(state: State): string;

  /** 可选：一次运行开始时注入随机数来源（确定性的找红球用不到） */
  useRandom?(rng: RandomSource): void;
}
