// @vitest-environment happy-dom
import { mount, type VueWrapper } from '@vue/test-utils';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { nextTick } from 'vue';

import App from '../src/App.vue';

/**
 * 界面接线测试：确认"点按钮 → 球真的动起来"这条链路是通的。
 * requestAnimationFrame 换成手动泵，让 50ms 一步的逻辑完全可控、可复现。
 */

let frameQueue: FrameRequestCallback[] = [];
let clock = 0;

beforeEach(() => {
  frameQueue = [];
  clock = 0;
  globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) => {
    frameQueue.push(callback);
    return frameQueue.length;
  }) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = (() => {
    frameQueue = [];
  }) as typeof cancelAnimationFrame;
});

afterEach(() => {
  frameQueue = [];
});

/** 推进若干帧（默认 16ms 一帧） */
function pump(frames: number, deltaMs = 16): void {
  for (let i = 0; i < frames; i += 1) {
    clock += deltaMs;
    const pending = frameQueue;
    frameQueue = [];
    for (const callback of pending) callback(clock);
  }
}

function button(wrapper: VueWrapper, label: string) {
  const found = wrapper.findAll('button').find((item) => item.text() === label);
  if (!found) throw new Error(`找不到按钮：${label}`);
  return found;
}

const steps = (wrapper: VueWrapper) => Number(wrapper.get('[data-testid="steps"]').text());
const ballPosition = (wrapper: VueWrapper) => {
  const style = wrapper.get('.agent').attributes('style') ?? '';
  return style;
};

let wrapper: VueWrapper | null = null;
afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

describe('界面', () => {
  it('渲染 4x4 网格、3 个陷阱、1 个终点，球在起点', () => {
    wrapper = mount(App);
    expect(wrapper.findAll('.cell')).toHaveLength(16);
    expect(wrapper.findAll('.cell--trap')).toHaveLength(3);
    expect(wrapper.findAll('.goal')).toHaveLength(1);
    expect(ballPosition(wrapper)).toContain('left: 0%');
    expect(ballPosition(wrapper)).toContain('top: 0%');
    expect(wrapper.get('[data-testid="algorithm"]').text()).toBe('未运行');
    expect(wrapper.get('[data-testid="state-count"]').text()).toBe('0 / 16');
  });

  it('点 Q-Learning start 之后球会动、步数与回合数会涨', async () => {
    wrapper = mount(App);
    const positionBefore = ballPosition(wrapper);
    await button(wrapper, 'Q-Learning start').trigger('click');
    expect(wrapper.get('[data-testid="algorithm"]').text()).toBe('Q-Learning');
    expect(steps(wrapper)).toBe(0);

    pump(20); // 20 帧 x 16ms = 320ms，按 50ms 一步应该走了 6 步
    await nextTick();

    expect(steps(wrapper)).toBeGreaterThanOrEqual(5);
    // 才走了几步，Q 表还不可能覆盖全部 16 个状态，只要它开始长了就说明学习在进行
    const stateCount = Number(wrapper.get('[data-testid="state-count"]').text().split(' / ')[0]);
    expect(stateCount).toBeGreaterThan(0);
    expect(stateCount).toBeLessThanOrEqual(16);
    expect(ballPosition(wrapper)).not.toBe(positionBefore);
  });

  it('球撞到终点或陷阱后会从起点重开一局（回合数 +1）', async () => {
    wrapper = mount(App);
    await button(wrapper, 'Q-Learning start').trigger('click');

    pump(200); // 3.2 秒，足够跑完好几集
    await nextTick();

    expect(Number(wrapper.get('[data-testid="episode"]').text())).toBeGreaterThan(0);
  });

  it('stop 之后不再推进，且按钮变灰；reset 把球送回起点', async () => {
    wrapper = mount(App);
    await button(wrapper, 'Q-Learning start').trigger('click');
    pump(30);
    await nextTick();
    const stepsWhenStopped = steps(wrapper);

    await button(wrapper, 'stop').trigger('click');
    const stopButton = button(wrapper, 'stop');
    expect((stopButton.element as HTMLButtonElement).disabled).toBe(true);

    pump(60);
    await nextTick();
    expect(steps(wrapper)).toBe(stepsWhenStopped);

    await button(wrapper, "initialize blue ball's position").trigger('click');
    expect(ballPosition(wrapper)).toContain('left: 0%');
    expect(ballPosition(wrapper)).toContain('top: 0%');
  });

  it('Sarsa start 走的是另一套算法（按钮互不干扰）', async () => {
    wrapper = mount(App);
    await button(wrapper, 'Sarsa start').trigger('click');
    expect(wrapper.get('[data-testid="algorithm"]').text()).toBe('Sarsa');

    pump(20);
    await nextTick();
    expect(steps(wrapper)).toBeGreaterThanOrEqual(5);
  });
});
