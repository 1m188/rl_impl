<script setup lang="ts">
import { computed } from 'vue';

import type { Action } from '../domain/actions';
import type { FindBallEnv } from '../domain/env';
import {
  normalizeHeat,
  showsHeat,
  showsPolicy,
  type DisplayMode,
  type QTableView,
  type StateView,
} from '../domain/qTableView';

const props = defineProps<{
  env: FindBallEnv;
  ball: number;
  view: QTableView;
  displayMode: DisplayMode;
}>();

const ARROWS: Record<Action, string> = { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' };

interface Cell extends StateView {
  trap: boolean;
  goal: boolean;
  lastCol: boolean;
  lastRow: boolean;
  arrow: string;
  heat: string | undefined;
}

/**
 * 热力图配色：琥珀色，透明度随 Q 值上升。
 * 特意避开蓝色（球）与红色（终点），免得三者看混。
 */
function heatColor(cell: StateView): string | undefined {
  if (!showsHeat(props.displayMode)) return undefined;
  if (!cell.known || cell.terminal || cell.best === null) return undefined;
  const ratio = normalizeHeat(cell.best, props.view.min, props.view.max);
  return `rgba(255, 168, 26, ${(0.07 + ratio * 0.5).toFixed(3)})`;
}

const cells = computed<Cell[]>(() =>
  props.view.cells.map((cell) => ({
    ...cell,
    trap: props.env.traps.has(cell.state),
    goal: cell.state === props.env.goal,
    lastCol: cell.x === props.env.width - 1,
    lastRow: cell.y === props.env.height - 1,
    // 终点与陷阱不画策略：一集到这儿就结束了，它们的 Q 值恒为 0，画出来会误导
    arrow:
      showsPolicy(props.displayMode) && cell.known && !cell.terminal && cell.policy
        ? ARROWS[cell.policy]
        : '',
    heat: heatColor(cell),
  })),
);

/** 球按格子跳，不插值 —— 与原实现一样是"一格一格地走" */
const ballStyle = computed(() => ({
  left: `${(props.env.x(props.ball) / props.env.width) * 100}%`,
  top: `${(props.env.y(props.ball) / props.env.height) * 100}%`,
  width: `${100 / props.env.width}%`,
  height: `${100 / props.env.height}%`,
}));
</script>

<template>
  <div class="board-wrap">
    <div
      class="board"
      :style="{ '--cols': env.width, '--rows': env.height }"
      role="img"
      :aria-label="`${env.width} 乘 ${env.height} 的网格，蓝球在第 ${env.x(ball) + 1} 列第 ${env.y(ball) + 1} 行`"
    >
      <div
        v-for="cell in cells"
        :key="cell.state"
        class="cell"
        :class="{
          'cell--trap': cell.trap,
          'cell--last-col': cell.lastCol,
          'cell--last-row': cell.lastRow,
          'cell--unknown': !cell.known,
        }"
        :style="{ background: cell.heat }"
      >
        <span v-if="cell.goal" class="goal" />
        <span v-if="cell.arrow" class="arrow">{{ cell.arrow }}</span>
      </div>
      <div class="agent" :style="ballStyle">
        <span class="agent__dot" />
      </div>
    </div>

    <ul class="legend">
      <li><span class="swatch swatch--ball" />蓝球（当前状态）</li>
      <li><span class="swatch swatch--goal" />红球（终点，+10）</li>
      <li><span class="swatch swatch--trap" />黑块（陷阱，−10）</li>
      <li><span class="swatch swatch--heat" />底色 = max Q(s,·) 的相对高低</li>
      <li>箭头 = 该格当前的最优动作（argmax）</li>
    </ul>
  </div>
</template>

<style scoped>
.board-wrap {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* 尺寸与原 QWidget 的 400x400 一致，窄屏时按比例缩小 */
.board {
  position: relative;
  width: min(400px, 86vw);
  aspect-ratio: 1 / 1;
  display: grid;
  grid-template-columns: repeat(var(--cols), 1fr);
  grid-template-rows: repeat(var(--rows), 1fr);
  background: #fff;
}

.cell {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 只画内部网格线，与原 paintEvent 里的循环一致 */
.cell:not(.cell--last-col) {
  border-right: 1px solid #000;
}

.cell:not(.cell--last-row) {
  border-bottom: 1px solid #000;
}

/* 黑色陷阱方块：原实现是每格向内缩进 5px 的矩形 */
.cell--trap::after {
  content: '';
  position: absolute;
  inset: 5px;
  background: #000;
}

/* 红色终点：向内缩进 5px 的正圆 */
.goal {
  position: absolute;
  inset: 5px;
  border-radius: 50%;
  background: #e02020;
}

/* 策略箭头：压在热力底色之上 */
.arrow {
  position: relative;
  z-index: 1;
  font-size: 26px;
  line-height: 1;
  color: #6b4a00;
  text-shadow: 0 0 3px #fff, 0 0 3px #fff;
  pointer-events: none;
}

.agent {
  position: absolute;
  pointer-events: none;
  z-index: 2;
}

/* 蓝色球：同样向内缩进 5px */
.agent__dot {
  position: absolute;
  inset: 5px;
  border-radius: 50%;
  background: #1a4fd6;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.15);
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: 12px;
  color: #5a5a5a;
  max-width: min(400px, 86vw);
}

.legend li {
  display: flex;
  align-items: center;
  gap: 5px;
}

.swatch {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  display: inline-block;
}

.swatch--ball {
  background: #1a4fd6;
}

.swatch--goal {
  background: #e02020;
}

.swatch--trap {
  background: #000;
  border-radius: 2px;
}

.swatch--heat {
  background: rgba(255, 168, 26, 0.45);
  border-radius: 2px;
}
</style>
