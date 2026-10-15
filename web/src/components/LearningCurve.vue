<script setup lang="ts">
import { computed, ref } from 'vue';

import type { EpisodeRecord } from '../domain/trainer';

const props = defineProps<{ history: EpisodeRecord[] }>();

type Metric = 'steps' | 'reward';

const metric = ref<Metric>('steps');
/** 滑动平均的窗口：单集的抖动很大，平均值才看得出趋势 */
const AVERAGE_WINDOW = 20;

const WIDTH = 620;
const HEIGHT = 170;
const PAD_X = 40;
const PAD_Y = 16;

const series = computed(() =>
  props.history.map((record) => (metric.value === 'steps' ? record.steps : record.reward)),
);

const bounds = computed(() => {
  const values = series.value;
  if (values.length === 0) return { min: 0, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  // 步数从 0 起画更好看；收益是负数，就按实际最小值来
  if (metric.value === 'steps') min = Math.min(min, 0);
  if (min === max) max = min + 1;
  return { min, max };
});

function xOf(index: number, count: number): number {
  if (count <= 1) return PAD_X;
  return PAD_X + (index / (count - 1)) * (WIDTH - PAD_X * 2);
}

function yOf(value: number): number {
  const { min, max } = bounds.value;
  return HEIGHT - PAD_Y - ((value - min) / (max - min)) * (HEIGHT - PAD_Y * 2);
}

function toPath(values: readonly number[]): string {
  return values
    .map((value, index) => {
      const x = xOf(index, values.length);
      const y = yOf(value);
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

const rawPath = computed(() => toPath(series.value));

/** 滑动平均：i 处取 [i-19, i] 的平均（只看窗口内的数据，不跨越开始处） */
const averagePath = computed(() =>
  toPath(
    series.value.map((_, index) => {
      const from = Math.max(0, index - AVERAGE_WINDOW + 1);
      const window = series.value.slice(from, index + 1);
      return window.reduce((sum, value) => sum + value, 0) / window.length;
    }),
  ),
);

const yTicks = computed(() =>
  [bounds.value.max, (bounds.value.min + bounds.value.max) / 2, bounds.value.min].map((value) => ({
    value,
    y: yOf(value),
  })),
);

const formatTick = (value: number) =>
  metric.value === 'steps' ? String(Math.round(value)) : value.toFixed(1);

const episodeRange = computed(() => {
  if (props.history.length === 0) return '';
  const first = props.history[0].index;
  const last = props.history[props.history.length - 1].index;
  return first === last ? `第 ${first} 集` : `第 ${first} – ${last} 集`;
});

/** 最近 20 集的平均步数与平均收益，用来量化"学到什么程度了" */
const recent = computed(() => {
  const window = props.history.slice(-AVERAGE_WINDOW);
  if (window.length === 0) return null;
  const steps = window.reduce((sum, r) => sum + r.steps, 0) / window.length;
  const reward = window.reduce((sum, r) => sum + r.reward, 0) / window.length;
  const goals = window.filter((r) => r.end === 'goal').length;
  return { count: window.length, steps, reward, goals };
});
</script>

<template>
  <section class="curve">
    <header class="curve__header">
      <h2>学习曲线</h2>
      <div class="curve__controls">
        <button
          type="button"
          :class="{ active: metric === 'steps' }"
          @click="metric = 'steps'"
        >
          每集步数
        </button>
        <button
          type="button"
          :class="{ active: metric === 'reward' }"
          @click="metric = 'reward'"
        >
          每集累计收益
        </button>
      </div>
    </header>

    <p v-if="history.length === 0" class="curve__empty">
      还没有跑完一集。点 start 开始，或用"快进训练"直接跑几百集看收敛。
    </p>

    <template v-else>
      <svg :viewBox="`0 0 ${WIDTH} ${HEIGHT}`" class="curve__svg" role="img" aria-label="学习曲线">
        <line
          v-for="tick in yTicks"
          :key="`t-${tick.value}`"
          :x1="PAD_X"
          :x2="WIDTH - PAD_X / 2"
          :y1="tick.y"
          :y2="tick.y"
          class="grid"
        />
        <text
          v-for="tick in yTicks"
          :key="`l-${tick.value}`"
          :x="PAD_X - 6"
          :y="tick.y + 4"
          class="axis"
          text-anchor="end"
        >
          {{ formatTick(tick.value) }}
        </text>
        <path :d="rawPath" class="raw" />
        <path :d="averagePath" class="average" />
      </svg>

      <p class="curve__meta">
        <span>{{ episodeRange }}</span>
        <span>细线 = 每集实际值，粗线 = 近 {{ AVERAGE_WINDOW }} 集滑动平均</span>
        <span v-if="recent">
          最近 {{ recent.count }} 集：平均 {{ recent.steps.toFixed(1) }} 步、
          平均收益 {{ recent.reward.toFixed(1) }}、到终点 {{ recent.goals }}/{{ recent.count }} 次
        </span>
      </p>
    </template>
  </section>
</template>

<style scoped>
.curve {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  background: #fff;
  border: 1px solid #e2e2e2;
  border-radius: 10px;
}

.curve__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

h2 {
  margin: 0;
  font-size: 15px;
}

.curve__controls {
  display: flex;
  gap: 6px;
}

.curve__controls button {
  padding: 4px 10px;
  font: inherit;
  font-size: 12px;
  color: #1b1b1b;
  background: #f4f4f4;
  border: 1px solid #c9c9c9;
  border-radius: 999px;
  cursor: pointer;
}

.curve__controls button.active {
  color: #fff;
  background: #1a4fd6;
  border-color: #1a4fd6;
}

.curve__svg {
  width: 100%;
  height: auto;
  display: block;
}

.grid {
  stroke: #ececec;
  stroke-width: 1;
}

.axis {
  fill: #8a8a8a;
  font-size: 10px;
}

.raw {
  fill: none;
  stroke: #b9cdf5;
  stroke-width: 1.2;
}

.average {
  fill: none;
  stroke: #1a4fd6;
  stroke-width: 2;
  stroke-linejoin: round;
}

.curve__empty,
.curve__meta {
  margin: 0;
  font-size: 12px;
  color: #6b6b6b;
}

.curve__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
}
</style>
