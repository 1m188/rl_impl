// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { VueWrapper } from '@vue/test-utils';

import {
  activeTab,
  arrowCount,
  ballCell,
  cellLabel,
  button,
  click,
  episodeOf,
  heatCount,
  installFrameStub,
  mountApp,
  nextTick,
  pump,
  pumpUntilBallMoves,
  stateCountOf,
  stepsOf,
  textOf,
} from './helpers/ui';

let wrapper: VueWrapper | null = null;

beforeEach(() => {
  installFrameStub();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe('界面：基本接线', () => {
  it('渲染 4x4 网格、3 个陷阱、1 个终点，球在起点', () => {
    wrapper = mountApp();
    expect(activeTab(wrapper).findAll('.cell')).toHaveLength(16);
    expect(activeTab(wrapper).findAll('.cell--trap')).toHaveLength(3);
    expect(activeTab(wrapper).findAll('.goal')).toHaveLength(1);
    expect(ballCell(wrapper)).toStrictEqual({ x: 0, y: 0 });
    expect(textOf(wrapper, 'algorithm')).toBe('未运行');
    expect(textOf(wrapper, 'state-count')).toBe('0 / 16');
    // 还没开始学，棋盘上不应该有任何叠加信息
    expect(arrowCount(wrapper)).toBe(0);
    expect(heatCount(wrapper)).toBe(0);
  });

  it('点 Q-Learning start 之后球会动、步数与 Q 表都会长', async () => {
    wrapper = mountApp();
    const positionBefore = cellLabel(wrapper);
    await click(wrapper, 'Q-Learning start');
    expect(textOf(wrapper, 'algorithm')).toBe('Q-Learning');
    expect(stepsOf(wrapper)).toBe(0);

    pump(20); // 20 帧 x 16ms = 320ms，按 50ms 一步应该走了 6 步
    await nextTick();

    expect(stepsOf(wrapper)).toBeGreaterThanOrEqual(5);
    expect(stateCountOf(wrapper)).toBeGreaterThan(0);
    expect(cellLabel(wrapper)).not.toBe(positionBefore);
  });

  it('球撞到终点或陷阱后会从起点重开一局（回合数 +1）', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(200);
    await nextTick();
    expect(episodeOf(wrapper)).toBeGreaterThan(0);
  });

  it('Sarsa start 走的是另一套算法（按钮互不干扰）', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Sarsa start');
    expect(textOf(wrapper, 'algorithm')).toBe('Sarsa');

    pump(20);
    await nextTick();
    expect(stepsOf(wrapper)).toBeGreaterThanOrEqual(5);
  });

  it('stop 结束这一轮并清空统计与 Q 表（与原实现一样丢弃学习结果）', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(30);
    await nextTick();
    expect(stepsOf(wrapper)).toBeGreaterThan(0);

    await click(wrapper, 'stop');
    expect((button(wrapper, 'stop').element as HTMLButtonElement).disabled).toBe(true);
    expect(textOf(wrapper, 'run-state')).toBe('空闲');
    expect(textOf(wrapper, 'algorithm')).toBe('未运行');
    expect(stepsOf(wrapper)).toBe(0);
    expect(episodeOf(wrapper)).toBe(0);
    expect(stateCountOf(wrapper)).toBe(0);
    expect(arrowCount(wrapper)).toBe(0);
    expect(ballCell(wrapper)).toStrictEqual({ x: 0, y: 0 });

    // 停下之后继续泵帧也不该有任何推进
    pump(60);
    await nextTick();
    expect(stepsOf(wrapper)).toBe(0);
  });

  it('initialize blue ball\'s position 把球送回起点', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    expect(await pumpUntilBallMoves(wrapper)).toBe(true);

    await click(wrapper, "initialize blue ball's position");
    expect(ballCell(wrapper)).toStrictEqual({ x: 0, y: 0 });
  });
});
