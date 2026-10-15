/**
 * 动作定义 —— 对应原 findball.py 的 Action 枚举。
 *
 * 原实现把动作放进 Enum 再塞进 set，而 Enum 的 hash 取自成员名、字符串 hash 每个进程随机，
 * 于是动作的遍历顺序每次运行都不一样。这里改成显式有序数组（顺序照抄原 Enum 的声明顺序），
 * 好处是"同一份随机数序列 → 同一段学习轨迹"可以被测试复现。
 *
 * 注意：算法本身只把动作当作不透明的 key 使用（字典的键），不关心它的类型。
 */
export const ACTION_ORDER = ['UP', 'DOWN', 'LEFT', 'RIGHT'] as const;

export type Action = (typeof ACTION_ORDER)[number];

/** 与原 Action 枚举的值完全一致：(0,-1) (0,1) (-1,0) (1,0) */
export const DELTA: Readonly<Record<Action, readonly [number, number]>> = {
  UP: [0, -1],
  DOWN: [0, 1],
  LEFT: [-1, 0],
  RIGHT: [1, 0],
};

export function isAction(value: string): value is Action {
  return (ACTION_ORDER as readonly string[]).includes(value);
}
