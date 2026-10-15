// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { VueWrapper } from '@vue/test-utils';

import {
  activeTab,
  click,
  episodeOf,
  inactiveTextOf,
  installFrameStub,
  mountApp,
  nextTick,
  pump,
  snakeHead,
  snakeLength,
  snakeScore,
  stateCountOf,
  stepsOf,
  switchTab,
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

describe('贪吃蛇页签', () => {
  it('切到贪吃蛇会渲染 8x8 网格与长度 3 的蛇；默认还是找红球', async () => {
    wrapper = mountApp();
    // 默认停在找红球
    expect(activeTab(wrapper).findAll('.agent')).toHaveLength(1);

    await switchTab(wrapper, '贪吃蛇');
    expect(activeTab(wrapper).findAll('.cell')).toHaveLength(64);
    expect(activeTab(wrapper).findAll('.segment')).toHaveLength(3);
    expect(snakeLength(wrapper)).toBe(3);
    expect(snakeScore(wrapper)).toBe(0);
    expect(snakeHead(wrapper)).toBe('4,4');
    expect(textOf(wrapper, 'algorithm')).toBe('未运行');
  });

  it('贪吃蛇的默认超参比找红球更爱探索（ε=0.1, γ=0.9）', async () => {
    wrapper = mountApp();
    await switchTab(wrapper, '贪吃蛇');

    const inputs = activeTab(wrapper).findAll('input[type="number"]');
    expect(inputs.map((input) => (input.element as HTMLInputElement).value)).toStrictEqual([
      '0.5',
      '0.9',
      '0.1',
    ]);
  });

  it('点 Q-Learning start 之后蛇会动、步数会涨', async () => {
    wrapper = mountApp();
    await switchTab(wrapper, '贪吃蛇');
    const headBefore = snakeHead(wrapper);

    await click(wrapper, 'Q-Learning start');
    expect(textOf(wrapper, 'algorithm')).toBe('Q-Learning');
    expect(stepsOf(wrapper)).toBe(0);

    // 逐帧看蛇头有没有离开过起点：开局 Q 表全是 0，蛇会一路向上撞墙，
    // 换局之后又会回到起点，所以不能只看"某一时刻的位置"。
    let moved = false;
    for (let i = 0; i < 12 && !moved; i += 1) {
      pump(1);
      await nextTick();
      moved = snakeHead(wrapper) !== headBefore;
    }

    expect(moved).toBe(true);
    expect(stepsOf(wrapper)).toBeGreaterThan(0);
  });

  it('快进 100 集之后回合数、Q 表与曲线都到位', async () => {
    wrapper = mountApp();
    await switchTab(wrapper, '贪吃蛇');
    await click(wrapper, 'Q-Learning start');
    await click(wrapper, '快进 100 集');

    expect(episodeOf(wrapper)).toBe(100);
    expect(stepsOf(wrapper)).toBeGreaterThan(100);
    expect(stateCountOf(wrapper)).toBeGreaterThan(20);
    expect(activeTab(wrapper).findAll('svg path').length).toBeGreaterThanOrEqual(2);
    expect(activeTab(wrapper).get('.curve__meta').text()).toContain('最近 20 集');
  });

  it('AI 视角面板显示蛇头周围 8 格的危险位、食物方位与当前状态键', async () => {
    wrapper = mountApp();
    await switchTab(wrapper, '贪吃蛇');
    await click(wrapper, 'Q-Learning start');

    const panel = activeTab(wrapper).get('.state-panel');
    // 开局：西边是自己身体 -> 1 个危险格
    expect(panel.findAll('.mini__cell--danger')).toHaveLength(1);
    expect(panel.findAll('.mini__cell--head')).toHaveLength(1);
    expect(panel.findAll('.mini__cell--food')).toHaveLength(1);
    expect(panel.text()).toContain('状态键');

    // Q 表里还没有这个状态时的提示
    expect(panel.text()).toContain('还没进过 Q 表');

    await click(wrapper, '快进 100 集');
    // 跑过之后这个状态已经进过表：价值条只会出现 3 个动作 ——
    // 蛇不能掉头，所以 Q 表里这一行就只有 3 个合法动作
    expect(activeTab(wrapper).get('.state-panel').findAll('.values li').length).toBe(3);
  });

  it('切走的页签会停帧但保留状态，切回来继续', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(20);
    await nextTick();
    const stepsBeforeSwitch = stepsOf(wrapper);
    expect(stepsBeforeSwitch).toBeGreaterThan(0);

    await switchTab(wrapper, '贪吃蛇');
    pump(40);
    await nextTick();
    // 找红球页签已经不在激活状态，步数不该再涨
    expect(inactiveTextOf(wrapper, 'steps')).toBe(String(stepsBeforeSwitch));

    await switchTab(wrapper, '找红球');
    expect(stepsOf(wrapper)).toBe(stepsBeforeSwitch);
    pump(20);
    await nextTick();
    expect(stepsOf(wrapper)).toBeGreaterThan(stepsBeforeSwitch);
  });

  it('两个页签各跑各的，互不影响', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(10);
    await nextTick();

    await switchTab(wrapper, '贪吃蛇');
    expect(textOf(wrapper, 'algorithm')).toBe('未运行');
    expect(stepsOf(wrapper)).toBe(0);

    await click(wrapper, 'Sarsa start');
    pump(10);
    await nextTick();
    expect(textOf(wrapper, 'algorithm')).toBe('Sarsa');
    expect(stepsOf(wrapper)).toBeGreaterThan(0);

    // 切回找红球，它自己的那次运行还在
    await switchTab(wrapper, '找红球');
    expect(textOf(wrapper, 'algorithm')).toBe('Q-Learning');
  });
});
