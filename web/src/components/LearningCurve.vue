<script setup lang="ts">
import { computed, ref } from 'vue';

import type { EpisodeRecord } from '../domain/trainer';
import { endReasonLabel } from '../ui/labels';

const props = defineProps<{ history: EpisodeRecord[] }>();

type Metric = 'steps' | 'reward';

const METRICS: ReadonlyArray<{ value: Metric; label: string }> = [
  { value: 'steps', label: '每集步数' },
  { value: 'reward', label: '每集累计收益' },
];

const metric = ref<Metric>('steps');
const hoverIndex = ref<number | null>(null);

/** 滑动平均的窗口：单集的抖动很大，平均值才看得出趋势 */
const AVERAGE_WINDOW = 20;

const WIDTH = 640;
const HEIGHT = 190;
const PAD_X = 44;
const PAD_Y = 18;

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

const average = computed(() =>
  series.value.map((_, index) => {
    const from = Math.max(0, index - AVERAGE_WINDOW + 1);
    const window = series.value.slice(from, index + 1);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  }),
);

function linePath(values: readonly number[]): string {
  return values
    .map((value, index) => {
      const x = xOf(index, values.length);
      const y = yOf(value);
      return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

const rawPath = computed(() => linePath(series.value));
const averagePath = computed(() => linePath(average.value));

/** 平均线下方填一层渐变，让曲线有"体量" */
const areaPath = computed(() => {
  const values = average.value;
  if (values.length === 0) return '';
  const base = HEIGHT - PAD_Y;
  const points = values.map((value, index) => [xOf(index, values.length), yOf(value)] as const);
  const line = points.map(([x, y]) => `L${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const first = points[0];
  const last = points[points.length - 1];
  return `M${first[0].toFixed(1)},${base} ${line} L${last[0].toFixed(1)},${base} Z`;
});

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

/** 最近 20 集的平均步数、平均收益与收尾方式分布 */
const recent = computed(() => {
  const window = props.history.slice(-AVERAGE_WINDOW);
  if (window.length === 0) return null;
  const steps = window.reduce((sum, r) => sum + r.steps, 0) / window.length;
  const reward = window.reduce((sum, r) => sum + r.reward, 0) / window.length;

  const counts = new Map<string, number>();
  for (const record of window) counts.set(record.end, (counts.get(record.end) ?? 0) + 1);
  const reasons = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([reason, count]) => ({ reason, label: endReasonLabel(reason), count }));

  return { count: window.length, steps, reward, reasons };
});

const metricIndex = computed(() => METRICS.findIndex((item) => item.value === metric.value));

/** 鼠标在图上移动时，找出最近的那一集 */
function onMove(event: MouseEvent): void {
  const svg = event.currentTarget as SVGSVGElement;
  const rect = svg.getBoundingClientRect();
  const count = series.value.length;
  if (rect.width === 0 || count === 0) return;
  if (count === 1) {
    hoverIndex.value = 0;
    return;
  }
  const x = ((event.clientX - rect.left) / rect.width) * WIDTH;
  const ratio = (x - PAD_X) / (WIDTH - PAD_X * 2);
  hoverIndex.value = Math.max(0, Math.min(count - 1, Math.round(ratio * (count - 1))));
}

const hovered = computed(() => {
  const index = hoverIndex.value;
  if (index === null || index >= series.value.length) return null;
  const value = series.value[index];
  const avg = average.value[index];
  const record = props.history[index];
  const x = xOf(index, series.value.length);
  const flip = x > WIDTH * 0.72;
  return {
    x,
    y: yOf(value),
    value,
    average: avg,
    episode: record.index,
    end: record.end,
    anchor: flip ? 'end' : 'start',
    labelX: flip ? x - 10 : x + 10,
  };
});
</script>

<template>
  <section class="card curve" :style="{ '--i': 2 }">
    <header class="card__head">
      <h2 class="card__title">学习曲线</h2>
      <div class="seg" :style="{ '--count': METRICS.length, '--index': metricIndex }">
        <span class="seg__thumb" />
        <button
          v-for="item in METRICS"
          :key="item.value"
          type="button"
          class="seg__item"
          :class="{ active: metric === item.value }"
          @click="metric = item.value"
        >
          {{ item.label }}
        </button>
      </div>
    </header>

    <p v-if="history.length === 0" class="empty">
      还没有跑完一集。点 start 开始，或用"快进训练"直接跑几百集看收敛。
    </p>

    <template v-else>
      <svg
        :viewBox="`0 0 ${WIDTH} ${HEIGHT}`"
        class="chart"
        role="img"
        aria-label="学习曲线"
        @mousemove="onMove"
        @mouseleave="hoverIndex = null"
      >
        <defs>
          <linearGradient id="curve-area" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgba(34, 211, 238, 0.38)" />
            <stop offset="100%" stop-color="rgba(34, 211, 238, 0)" />
          </linearGradient>
          <filter id="curve-glow" x="-20%" y="-40%" width="140%" height="180%">
            <feGaussianBlur stdDeviation="3.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

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
          :x="PAD_X - 8"
          :y="tick.y + 4"
          class="axis"
          text-anchor="end"
        >
          {{ formatTick(tick.value) }}
        </text>

        <path :d="areaPath" class="area" />
        <path :d="rawPath" class="raw" />
        <path :key="metric" :d="averagePath" class="average" />

        <template v-if="hovered">
          <line :x1="hovered.x" :x2="hovered.x" :y1="PAD_Y" :y2="HEIGHT - PAD_Y" class="crosshair" />
          <circle :cx="hovered.x" :cy="hovered.y" r="4" class="dot" />
          <text
            :x="hovered.labelX"
            :y="Math.max(14, hovered.y - 10)"
            class="readout"
            :text-anchor="hovered.anchor"
          >
            第 {{ hovered.episode }} 集 · {{ metric === 'steps' ? `${hovered.value} 步` : hovered.value.toFixed(1) }}
            · 均值 {{ hovered.average.toFixed(1) }} · {{ endReasonLabel(hovered.end) }}
          </text>
        </template>
      </svg>

      <p class="curve__meta">
        <span>{{ episodeRange }}</span>
        <span>细线 = 每集实际值，粗线 = 近 {{ AVERAGE_WINDOW }} 集滑动平均</span>
        <span v-if="recent">
          最近 {{ recent.count }} 集：平均 {{ recent.steps.toFixed(1) }} 步、平均收益
          {{ recent.reward.toFixed(1) }}、收尾
          <template v-for="(item, index) in recent.reasons" :key="item.reason"
            >{{ index > 0 ? ' / ' : '' }}{{ item.label }} {{ item.count }}</template
          >
        </span>
      </p>
    </template>
  </section>
</template>

<style scoped>
.curve {
  display: flex;
  flex-direction: column;
}

.empty {
  padding: 28px 0;
  text-align: center;
  font-size: 12px;
  color: var(--text-dim);
}

.chart {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 10px;
  background: radial-gradient(120% 120% at 50% 0%, rgba(34, 211, 238, 0.06), transparent 70%);
}

.grid {
  stroke: rgba(255, 255, 255, 0.07);
  stroke-width: 1;
  stroke-dasharray: 3 5;
}

.axis {
  fill: var(--text-dim);
  font-size: 10px;
  font-family: var(--font-mono);
}

.area {
  fill: url(#curve-area);
}

.raw {
  fill: none;
  stroke: rgba(148, 163, 184, 0.5);
  stroke-width: 1.1;
}

.average {
  fill: none;
  stroke: var(--accent);
  stroke-width: 2.2;
  stroke-linejoin: round;
  stroke-linecap: round;
  filter: url(#curve-glow);
  animation: draw 900ms var(--ease-out) both;
}

@keyframes draw {
  from {
    stroke-dasharray: 2400;
    stroke-dashoffset: 2400;
  }
  to {
    stroke-dasharray: 2400;
    stroke-dashoffset: 0;
  }
}

.crosshair {
  stroke: rgba(255, 255, 255, 0.35);
  stroke-width: 1;
  stroke-dasharray: 4 4;
}

.dot {
  fill: var(--accent);
  stroke: #04121a;
  stroke-width: 1.5;
}

.readout {
  fill: var(--text);
  font-size: 11px;
  font-family: var(--font-mono);
}

.curve__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-dim);
}
</style>
