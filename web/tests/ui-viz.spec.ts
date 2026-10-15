// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { VueWrapper } from '@vue/test-utils';

import {
  arrowCount,
  cellLabel,
  click,
  episodeOf,
  heatCount,
  installFrameStub,
  mountApp,
  nextTick,
  pump,
  stateCountOf,
  stepsOf,
  textOf,
  trailCount,
} from './helpers/ui';

let wrapper: VueWrapper | null = null;

beforeEach(() => {
  installFrameStub();
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

/** 设好种子再开始，保证轨迹可复现 */
async function startWithSeed(target: VueWrapper, seed: string, kind = 'Q-Learning start') {
  const input = target.get('input[placeholder="留空 = 随机"]');
  await input.setValue(seed);
  await click(target, kind);
}

describe('界面：可视化与训练控制', () => {
  it('pause 保留 Q 表但不再推进，resume 继续', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(30);
    await nextTick();
    const paused = stepsOf(wrapper);
    const known = stateCountOf(wrapper);
    expect(paused).toBeGreaterThan(0);

    await click(wrapper, 'pause');
    expect(textOf(wrapper, 'run-state')).toBe('已暂停');
    pump(60);
    await nextTick();
    expect(stepsOf(wrapper)).toBe(paused);
    // 关键区别：pause 不像 stop 那样把学到的东西丢掉
    expect(stateCountOf(wrapper)).toBe(known);

    await click(wrapper, 'resume');
    pump(20);
    await nextTick();
    expect(stepsOf(wrapper)).toBeGreaterThan(paused);
  });

  it('暂停时 step 只走一步', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(20);
    await nextTick();
    await click(wrapper, 'pause');

    const before = stepsOf(wrapper);
    await click(wrapper, 'step');
    expect(stepsOf(wrapper)).toBe(before + 1);
    await click(wrapper, 'step');
    expect(stepsOf(wrapper)).toBe(before + 2);
  });

  it('倍速真的改变推进速度（5x 在同样的帧数里走更多步）', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(30);
    await nextTick();
    const atOneX = stepsOf(wrapper);

    await click(wrapper, '5x');
    const beforeFast = stepsOf(wrapper);
    pump(30);
    await nextTick();
    const fastDelta = stepsOf(wrapper) - beforeFast;

    // 30 帧 x 16ms = 480ms：1x（50ms/步）约 9 步，5x（10ms/步）约 48 步
    expect(atOneX).toBeLessThan(fastDelta);
    expect(fastDelta).toBeGreaterThan(30);
  });

  it('快进 100 集：回合数与学习曲线数据立刻到位', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    await click(wrapper, '快进 100 集');

    expect(episodeOf(wrapper)).toBe(100);
    expect(stepsOf(wrapper)).toBeGreaterThan(100);
    // 学习曲线画出来了（有折线路径）
    expect(wrapper.findAll('svg path').length).toBeGreaterThanOrEqual(2);
    expect(wrapper.get('.curve__meta').text()).toContain('最近 20 集');
  });

  it('训练之后棋盘上出现策略箭头与热力底色，切到"不显示"就消失', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    expect(arrowCount(wrapper)).toBe(0);

    await click(wrapper, '快进 100 集');
    // 12 个非终止格子（16 格去掉终点与 3 个陷阱）都应该学到动作了
    expect(arrowCount(wrapper)).toBe(12);
    expect(heatCount(wrapper)).toBe(12);
    // 终点与陷阱不画策略（它们的 Q 值恒为 0，画出来会误导）
    expect(wrapper.find('.cell--trap .arrow').exists()).toBe(false);

    await click(wrapper, '不显示');
    expect(arrowCount(wrapper)).toBe(0);
    expect(heatCount(wrapper)).toBe(0);

    await click(wrapper, '只看策略箭头');
    expect(arrowCount(wrapper)).toBe(12);
    expect(heatCount(wrapper)).toBe(0);
  });

  it('同一个种子跑出同一条轨迹（可复现），不同种子则不同', async () => {
    const trace = async (seed: string) => {
      const app = mountApp();
      await startWithSeed(app, seed);
      const positions: string[] = [];
      for (let i = 0; i < 120; i += 1) {
        pump(1, 100); // 每帧 100ms -> 每帧走 2 步
        await nextTick();
        positions.push(cellLabel(app));
      }
      const summary = `${stepsOf(app)}/${episodeOf(app)}/${stateCountOf(app)}`;
      app.unmount();
      return { positions, summary };
    };

    const a = await trace('12345');
    const b = await trace('12345');
    expect(b.positions).toStrictEqual(a.positions);
    expect(b.summary).toBe(a.summary);

    // 种子不同 -> 至少有一处不一样（ε=0.01，240 步里必然出现过探索）
    const c = await trace('999');
    expect(c.positions).not.toStrictEqual(a.positions);
  });

  it('种子填了非数字会提示按随机处理', async () => {
    wrapper = mountApp();
    const input = wrapper.get('input[placeholder="留空 = 随机"]');
    await input.setValue('abc');
    expect(wrapper.get('.warn').text()).toContain('按随机处理');
  });

  it('Q 表数值表：训练后列出 16 行，并高亮每格的最优动作', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    await click(wrapper, '快进 100 集');

    const rows = wrapper.findAll('.qtable tbody tr');
    expect(rows).toHaveLength(16);
    expect(wrapper.findAll('.qtable td.best').length).toBeGreaterThanOrEqual(12);
    // 终点与陷阱那几行的"最优"列应该是空的
    expect(wrapper.findAll('.qtable td.policy').filter((td) => td.text() === '—').length).toBe(4);
  });

  it('球走过之后留下轨迹残影，stop 时清空', async () => {
    wrapper = mountApp();
    expect(trailCount(wrapper)).toBe(0);

    await click(wrapper, 'Q-Learning start');
    pump(8); // 约 1-2 步，不会刚好赶上换局
    await nextTick();
    expect(trailCount(wrapper)).toBeGreaterThan(0);

    await click(wrapper, 'stop');
    expect(trailCount(wrapper)).toBe(0);
  });

  it('一集结束时会在终点或陷阱上留下命中标记（只标记刚结束的那一格）', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    pump(200);
    await nextTick();
    expect(episodeOf(wrapper)).toBeGreaterThan(0);

    const marks = wrapper.findAll('.ring').length + wrapper.findAll('.trap--hit').length;
    expect(marks).toBe(1);
  });

  it('鼠标移到格子上会显示该格的四个动作价值', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    await click(wrapper, '快进 100 集');

    expect(wrapper.get('.inspect').text()).toContain('把鼠标移');

    const cells = wrapper.findAll('.cell');
    await cells[0].trigger('mouseenter'); // 行优先，index 0 = (0,0)
    await nextTick();
    const readout = wrapper.get('.inspect').text();
    expect(readout).toContain('(0, 0)');
    expect(readout).toContain('下'); // 四个动作的中文标签
    expect(readout).toContain('右');

    await cells[5].trigger('mouseenter'); // index 5 = (1,1)，是陷阱
    await nextTick();
    expect(wrapper.get('.inspect').text()).toContain('陷阱');
  });

  it('单步细节面板会显示最近一步的 s/a/r/s\' 与 Q 值变化', async () => {
    wrapper = mountApp();
    await click(wrapper, 'Q-Learning start');
    await click(wrapper, '快进 100 集');

    const chain = wrapper.get('.chain').findAll('li').map((li) => li.text());
    expect(chain).toHaveLength(4);
    expect(chain[0]).toMatch(/^s\(/);
    // 奖励是负的距离塑形、-10（陷阱）或 +10（终点），取决于最后一步落在哪
    expect(chain[2]).toMatch(/^r-?\d/);
    expect(wrapper.get('.update').text()).toContain('→');
  });
});
