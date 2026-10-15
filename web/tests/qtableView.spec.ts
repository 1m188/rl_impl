import { describe, expect, it } from 'vitest';

import { ACTION_ORDER, type Action } from '../src/domain/actions';
import { createFindBallEnv } from '../src/domain/env';
import { QLearning } from '../src/domain/qLearning';
import {
  averageBest,
  normalizeHeat,
  readQTableView,
  showsHeat,
  showsPolicy,
} from '../src/domain/qTableView';
import { scriptedSource } from '../src/domain/rng';

const env = createFindBallEnv();

/** 手工构造一个 Q 表，方便精确断言（不跑算法，纯视图测试） */
function makeAgent(rows: Record<number, Partial<Record<Action, number>>>): QLearning {
  const agent = new QLearning(0.5, 0.5, 0.01, scriptedSource([]));
  for (const [key, values] of Object.entries(rows)) {
    const row = new Map<Action, number>();
    for (const action of ACTION_ORDER) {
      const value = values[action];
      if (value !== undefined) row.set(action, value);
    }
    agent.qtable.set(Number(key), row);
  }
  return agent;
}

const cellAt = (view: ReturnType<typeof readQTableView>, state: number) => {
  const cell = view.cells.find((item) => item.state === state);
  if (!cell) throw new Error(`视图里没有状态 ${state}`);
  return cell;
};

describe('Q 表视图', () => {
  it('没有 agent 时：16 个格子都在、都标记为未知、取值范围为 0', () => {
    const view = readQTableView(null, env);
    expect(view.cells).toHaveLength(16);
    expect(view.knownCount).toBe(0);
    expect(view.stateCount).toBe(16);
    expect(view.min).toBe(0);
    expect(view.max).toBe(0);
    expect(view.cells.every((cell) => !cell.known && cell.policy === null)).toBe(true);
  });

  it('标记终点与陷阱为终止状态（它们不参与策略可视化）', () => {
    const view = readQTableView(null, env);
    const terminals = view.cells.filter((cell) => cell.terminal).map((cell) => cell.state);
    expect(terminals).toHaveLength(4);
    expect(terminals).toContain(env.goal);
    for (const trap of env.traps) expect(terminals).toContain(trap);
  });

  it('policy 取 argmax；并列时取动作顺序靠前的那个', () => {
    // 状态 0 是 (0,0)，合法动作只有 DOWN / RIGHT
    const view = readQTableView(makeAgent({ 0: { DOWN: 1, RIGHT: 3 } }), env);
    expect(cellAt(view, 0).policy).toBe('RIGHT');
    expect(cellAt(view, 0).best).toBe(3);
    expect(cellAt(view, 0).worst).toBe(1);

    const tied = readQTableView(makeAgent({ 0: { DOWN: 2, RIGHT: 2 } }), env);
    expect(cellAt(tied, 0).policy).toBe('DOWN'); // ACTION_ORDER 里 DOWN 在前
  });

  it('min / max 只统计已知状态，热力归一化也就只按它们来', () => {
    const view = readQTableView(
      makeAgent({ 0: { DOWN: -2, RIGHT: -1 }, 1: { LEFT: 4, RIGHT: 5, DOWN: 0 } }),
      env,
    );
    expect(view.knownCount).toBe(2);
    expect(view.min).toBe(-1); // 状态 0 的 best
    expect(view.max).toBe(5); // 状态 1 的 best
    expect(cellAt(view, 0).known).toBe(true);
    expect(cellAt(view, 2).known).toBe(false);
  });

  it('normalizeHeat：端点夹紧，全部相等时取中值（避免除零）', () => {
    expect(normalizeHeat(0, 0, 10)).toBe(0);
    expect(normalizeHeat(10, 0, 10)).toBe(1);
    expect(normalizeHeat(5, 0, 10)).toBe(0.5);
    expect(normalizeHeat(-100, 0, 10)).toBe(0);
    expect(normalizeHeat(100, 0, 10)).toBe(1);
    expect(normalizeHeat(3, 3, 3)).toBe(0.5);
  });

  it('averageBest：已知状态 best 的平均值，没有已知状态时返回 null', () => {
    expect(averageBest(readQTableView(null, env))).toBeNull();
    const view = readQTableView(makeAgent({ 0: { DOWN: 0, RIGHT: 4 }, 1: { LEFT: 2 } }), env);
    expect(averageBest(view)).toBe(3); // (4 + 2) / 2
  });

  it('显示模式的组合开关', () => {
    expect([showsPolicy('both'), showsHeat('both')]).toStrictEqual([true, true]);
    expect([showsPolicy('policy'), showsHeat('policy')]).toStrictEqual([true, false]);
    expect([showsPolicy('heat'), showsHeat('heat')]).toStrictEqual([false, true]);
    expect([showsPolicy('none'), showsHeat('none')]).toStrictEqual([false, false]);
  });
});
