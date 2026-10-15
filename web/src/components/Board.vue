<script setup lang="ts">
import { computed } from 'vue';

import type { FindBallEnv } from '../domain/env';

const props = defineProps<{
  env: FindBallEnv;
  ball: number;
}>();

interface Cell {
  state: number;
  trap: boolean;
  goal: boolean;
  lastCol: boolean;
  lastRow: boolean;
}

const cells = computed<Cell[]>(() => {
  const list: Cell[] = [];
  for (let y = 0; y < props.env.height; y += 1) {
    for (let x = 0; x < props.env.width; x += 1) {
      const state = props.env.pack(x, y);
      list.push({
        state,
        trap: props.env.traps.has(state),
        goal: state === props.env.goal,
        lastCol: x === props.env.width - 1,
        lastRow: y === props.env.height - 1,
      });
    }
  }
  return list;
});

/** 球按格子跳，不插值 —— 与原实现一样是"一格一格地走" */
const ballStyle = computed(() => ({
  left: `${(props.env.x(props.ball) / props.env.width) * 100}%`,
  top: `${(props.env.y(props.ball) / props.env.height) * 100}%`,
  width: `${100 / props.env.width}%`,
  height: `${100 / props.env.height}%`,
}));
</script>

<template>
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
      }"
    >
      <span v-if="cell.goal" class="goal" />
    </div>
    <div class="agent" :style="ballStyle">
      <span class="agent__dot" />
    </div>
  </div>
</template>

<style scoped>
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

.agent {
  position: absolute;
  pointer-events: none;
}

/* 蓝色球：同样向内缩进 5px */
.agent__dot {
  position: absolute;
  inset: 5px;
  border-radius: 50%;
  background: #1a4fd6;
}
</style>
