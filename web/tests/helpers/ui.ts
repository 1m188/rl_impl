import { mount, type VueWrapper } from '@vue/test-utils';
import { nextTick } from 'vue';

import App from '../../src/App.vue';

/**
 * 界面测试的公共工具。
 *
 * 页面现在有两个页签（找红球 / 贪吃蛇），两套面板同时在 DOM 里（只是隐藏），
 * 里面的按钮文字、data-testid 都一样，所以所有查询都限制在当前激活的页签内。
 *
 * requestAnimationFrame 换成手动泵：这样"50ms 一步"的逻辑完全可控，
 * 不用真的等时间，也不会受机器快慢影响。
 */

let frameQueue: FrameRequestCallback[] = [];
let clock = 0;

export function installFrameStub(): void {
  frameQueue = [];
  clock = 0;
  globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) => {
    frameQueue.push(callback);
    return frameQueue.length;
  }) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = (() => {
    frameQueue = [];
  }) as typeof cancelAnimationFrame;
}

/** 推进若干帧（默认 16ms 一帧） */
export function pump(frames: number, deltaMs = 16): void {
  for (let i = 0; i < frames; i += 1) {
    clock += deltaMs;
    const pending = frameQueue;
    frameQueue = [];
    for (const callback of pending) callback(clock);
  }
}

export function mountApp(): VueWrapper {
  return mount(App);
}

/** 当前激活页签的根元素 */
const scope = (wrapper: VueWrapper) => wrapper.get('[data-active="true"]');

/** 需要按整块查询（比如数格子）时用它，避免把隐藏页签里的元素也算进来 */
export const activeTab = (wrapper: VueWrapper) => scope(wrapper);

/** 读非激活页签里的某个 testid（用来验证"切走的页签状态被保留"） */
export const inactiveTextOf = (wrapper: VueWrapper, testid: string): string =>
  wrapper.get(`[data-active="false"] [data-testid="${testid}"]`).text();

/** 切换页签（页签按钮本身在激活区域之外，所以在整个 wrapper 里找） */
export async function switchTab(wrapper: VueWrapper, label: string): Promise<void> {
  const item = wrapper.findAll('.tabs__item').find((button) => button.text() === label);
  if (!item) throw new Error(`找不到页签：${label}`);
  await item.trigger('click');
  await nextTick();
}

/**
 * 按文字找按钮：优先精确匹配，找不到再退回"包含"匹配
 * （按钮上现在带了 ▶ ❚❚ 这类图标，text() 会把图标也算进去）。
 */
export function button(wrapper: VueWrapper, label: string) {
  const all = scope(wrapper).findAll('button');
  const exact = all.find((item) => item.text() === label);
  if (exact) return exact;
  const partial = all.find((item) => item.text().includes(label));
  if (partial) return partial;
  throw new Error(`找不到按钮：${label}`);
}

export async function click(wrapper: VueWrapper, label: string): Promise<void> {
  await button(wrapper, label).trigger('click');
  await nextTick();
}

export const textOf = (wrapper: VueWrapper, testid: string): string =>
  scope(wrapper).get(`[data-testid="${testid}"]`).text();

export const stepsOf = (wrapper: VueWrapper): number => Number(textOf(wrapper, 'steps'));

export const episodeOf = (wrapper: VueWrapper): number => Number(textOf(wrapper, 'episode'));

export const stateCountOf = (wrapper: VueWrapper): number =>
  Number(textOf(wrapper, 'state-count').split(' / ')[0]);

/** 找红球：球所在格子的坐标，直接读组件写好的 data-cell */
export function ballCell(wrapper: VueWrapper): { x: number; y: number } {
  const raw = scope(wrapper).get('.agent').attributes('data-cell') ?? '0,0';
  const [x, y] = raw.split(',').map(Number);
  return { x, y };
}

/** 位置的字符串形式，用来比较整条轨迹 */
export const cellLabel = (wrapper: VueWrapper): string =>
  scope(wrapper).get('.agent').attributes('data-cell') ?? '';

/** 一直泵帧直到球离开起点（最多 frames 帧），返回是否成功离开 */
export async function pumpUntilBallMoves(wrapper: VueWrapper, frames = 40): Promise<boolean> {
  for (let i = 0; i < frames; i += 1) {
    pump(1);
    await nextTick();
    const { x, y } = ballCell(wrapper);
    if (x !== 0 || y !== 0) return true;
  }
  return false;
}

export const arrowCount = (wrapper: VueWrapper): number =>
  scope(wrapper).findAll('.arrow').length;

/** 找红球：棋盘上有热力底色的格子数 */
export const heatCount = (wrapper: VueWrapper): number =>
  scope(wrapper)
    .findAll('.cell')
    .filter((cell) => (cell.attributes('style') ?? '').includes('background')).length;

/** 轨迹残影的点数 */
export const trailCount = (wrapper: VueWrapper): number =>
  scope(wrapper).findAll('.trail').length;

/** 贪吃蛇：蛇头坐标 / 当前长度 / 得分 */
export const snakeHead = (wrapper: VueWrapper): string =>
  scope(wrapper).get('[data-snake-head]').attributes('data-snake-head') ?? '';

export const snakeLength = (wrapper: VueWrapper): number =>
  Number(scope(wrapper).get('.board').attributes('data-length') ?? '0');

export const snakeScore = (wrapper: VueWrapper): number =>
  Number(scope(wrapper).get('.board').attributes('data-score') ?? '0');

export { nextTick };
