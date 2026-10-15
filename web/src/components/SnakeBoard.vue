<script setup lang="ts">
import { computed } from 'vue';

import type { SnakeEnv, SnakeFeatures, SnakeSnapshot } from '../domain/snake';

const props = defineProps<{
  env: SnakeEnv;
  snapshot: SnakeSnapshot;
  state: number;
  features: SnakeFeatures;
  burst: { key: number; state: number; reason: string } | null;
  moveDurationMs: number;
  teleported: boolean;
  running: boolean;
}>();

/** 背景网格（只为了画格线） */
const cells = computed(() => {
  const list: { key: number; lastCol: boolean; lastRow: boolean }[] = [];
  for (let y = 0; y < props.env.height; y += 1) {
    for (let x = 0; x < props.env.width; x += 1) {
      list.push({
        key: props.env.pack(x, y),
        lastCol: x === props.env.width - 1,
        lastRow: y === props.env.height - 1,
      });
    }
  }
  return list;
});

/** 蛇身：index 0 是蛇头。按 index 当 key，元素会被复用，位移就能靠 CSS 过渡连起来 */
const segments = computed(() =>
  props.snapshot.body.map((cell, index) => ({
    cell,
    index,
    isHead: index === 0,
    isTail: index === props.snapshot.body.length - 1,
    // 越靠尾巴越细、越淡，看起来像一条蛇而不是一串方块
    scale: Math.max(0.62, 1 - index * 0.045),
    opacity: Math.max(0.55, 1 - index * 0.05),
  })),
);

const transformOf = (cell: number) => ({
  transform: `translate(${props.env.x(cell) * 100}%, ${props.env.y(cell) * 100}%)`,
});

const headDirection = computed(() => props.snapshot.direction.toLowerCase());

const dying = computed(
  () => props.burst !== null && (props.burst.reason === 'death' || props.burst.reason === 'full'),
);
</script>

<template>
  <section class="card snake-card" :style="{ '--i': 0 }">
    <header class="card__head">
      <h2 class="card__title">棋盘</h2>
      <ul class="legend">
        <li><span class="swatch swatch--head" />蛇头</li>
        <li><span class="swatch swatch--body" />蛇身</li>
        <li><span class="swatch swatch--food" />食物 +10</li>
        <li>撞墙 / 咬到自己 −10</li>
      </ul>
    </header>

    <div
      class="board-wrap"
      :style="{
        '--cols': env.width,
        '--rows': env.height,
        '--move-duration': teleported ? '0ms' : `${moveDurationMs}ms`,
      }"
    >
      <div
        class="board"
        role="img"
        :data-length="snapshot.body.length"
        :data-score="snapshot.score"
        :aria-label="`${env.width} 乘 ${env.height} 的网格，蛇长 ${snapshot.body.length}，已吃 ${snapshot.score} 个食物`"
      >
        <div
          v-for="cell in cells"
          :key="cell.key"
          class="cell"
          :class="{ 'cell--last-col': cell.lastCol, 'cell--last-row': cell.lastRow }"
        />

        <span
          v-if="snapshot.food >= 0"
          :key="`food-${snapshot.food}-${snapshot.score}`"
          class="food"
          :style="transformOf(snapshot.food)"
        />

        <span
          v-for="segment in segments"
          :key="segment.index"
          class="segment"
          :class="{ 'segment--head': segment.isHead, [`segment--head-${headDirection}`]: segment.isHead }"
          :data-snake-head="
            segment.isHead ? `${env.x(segment.cell)},${env.y(segment.cell)}` : null
          "
          :style="{
            ...transformOf(segment.cell),
            '--scale': segment.scale,
            opacity: segment.opacity,
          }"
        >
          <span v-if="segment.isHead" class="eyes">
            <i /><i />
          </span>
        </span>

        <span v-if="dying" :key="`hit-${burst?.key}`" class="hit" />
      </div>

      <div class="inspect">
        <span class="inspect__pos num">
          长度 {{ snapshot.body.length }} · 得分 {{ snapshot.score }}
        </span>
        <span class="inspect__tag">
          朝向 {{ { UP: '上', DOWN: '下', LEFT: '左', RIGHT: '右' }[snapshot.direction] }} ·
          当前状态键 <span class="num">{{ state >= 0 ? state : '终止' }}</span>
        </span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.snake-card {
  display: flex;
  flex-direction: column;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: 11px;
  color: var(--text-muted);
}

.legend li {
  display: flex;
  align-items: center;
  gap: 5px;
}

.swatch {
  width: 10px;
  height: 10px;
  border-radius: 3px;
}

.swatch--head {
  background: linear-gradient(180deg, #7aa8ff, #1d4ed8);
  box-shadow: 0 0 8px rgba(37, 99, 235, 0.9);
}

.swatch--body {
  background: linear-gradient(180deg, #4f8ef7, #1e40af);
}

.swatch--food {
  background: radial-gradient(circle at 35% 30%, #ff8a8a, #d81f1f);
  border-radius: 50%;
  box-shadow: 0 0 8px rgba(224, 32, 32, 0.8);
}

.board-wrap {
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: center;
}

.board {
  position: relative;
  width: 100%;
  max-width: 440px;
  aspect-ratio: 1 / 1;
  display: grid;
  grid-template-columns: repeat(var(--cols), 1fr);
  grid-template-rows: repeat(var(--rows), 1fr);
  background: #fbfbfd;
  border-radius: 12px;
  overflow: hidden;
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.14),
    0 30px 60px -30px rgba(167, 139, 250, 0.55),
    0 10px 30px -20px rgba(0, 0, 0, 0.9);
}

.cell {
  position: relative;
}

.cell:not(.cell--last-col) {
  border-right: 1px solid rgba(0, 0, 0, 0.16);
}

.cell:not(.cell--last-row) {
  border-bottom: 1px solid rgba(0, 0, 0, 0.16);
}

/* 蛇身与蛇头：绝对定位 + transform 位移 */
.segment {
  position: absolute;
  top: 0;
  left: 0;
  width: calc(100% / var(--cols));
  height: calc(100% / var(--rows));
  pointer-events: none;
  z-index: 1;
  transition: transform var(--move-duration) var(--ease-out);
}

.segment::before {
  content: '';
  position: absolute;
  inset: 11%;
  border-radius: 26%;
  background: linear-gradient(160deg, #5b93f8, #1d4ed8);
  transform: scale(var(--scale));
  transition: transform var(--dur) var(--ease-out);
}

.segment--head {
  z-index: 3;
}

.segment--head::before {
  inset: 7%;
  border-radius: 30%;
  background: radial-gradient(circle at 34% 28%, #9ec2ff, #2563eb 62%, #123a9e);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.5),
    0 6px 16px -6px rgba(29, 78, 216, 0.95);
}

/* 眼睛按朝向摆位 */
.eyes {
  position: absolute;
  inset: 0;
}

.eyes i {
  position: absolute;
  width: 16%;
  height: 16%;
  border-radius: 50%;
  background: #f8fbff;
  box-shadow: 0 0 2px rgba(0, 0, 0, 0.4);
}

.segment--head-right .eyes i:first-child {
  right: 20%;
  top: 24%;
}

.segment--head-right .eyes i:last-child {
  right: 20%;
  bottom: 24%;
}

.segment--head-left .eyes i:first-child {
  left: 20%;
  top: 24%;
}

.segment--head-left .eyes i:last-child {
  left: 20%;
  bottom: 24%;
}

.segment--head-up .eyes i:first-child {
  top: 20%;
  left: 24%;
}

.segment--head-up .eyes i:last-child {
  top: 20%;
  right: 24%;
}

.segment--head-down .eyes i:first-child {
  bottom: 20%;
  left: 24%;
}

.segment--head-down .eyes i:last-child {
  bottom: 20%;
  right: 24%;
}

/* 食物：出现时弹一下 */
.food {
  position: absolute;
  top: 0;
  left: 0;
  width: calc(100% / var(--cols));
  height: calc(100% / var(--rows));
  pointer-events: none;
  z-index: 2;
  animation: food-pop 320ms var(--ease-spring) both;
}

.food::after {
  content: '';
  position: absolute;
  inset: 22%;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #ff9a9a, #d81f1f 70%);
  box-shadow: 0 0 14px rgba(216, 31, 31, 0.55);
  animation: food-pulse 1.6s ease-in-out infinite;
}

@keyframes food-pop {
  from {
    transform: scale(0.4);
    opacity: 0;
  }
}

@keyframes food-pulse {
  0%,
  100% {
    box-shadow: 0 0 10px rgba(216, 31, 31, 0.45);
  }
  50% {
    box-shadow: 0 0 20px rgba(216, 31, 31, 0.85);
  }
}

/* 撞死时整盘闪一下红光 */
.hit {
  position: absolute;
  inset: 0;
  z-index: 4;
  pointer-events: none;
  animation: hit-flash 620ms var(--ease-out) both;
}

@keyframes hit-flash {
  0% {
    background: rgba(251, 113, 133, 0.55);
    box-shadow: inset 0 0 60px rgba(251, 113, 133, 0.9);
  }
  100% {
    background: rgba(251, 113, 133, 0);
    box-shadow: inset 0 0 0 rgba(251, 113, 133, 0);
  }
}

.inspect {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 10px;
  min-height: 30px;
  width: 100%;
  max-width: 440px;
  padding: 6px 10px;
  font-size: 12px;
  color: var(--text-muted);
  background: rgba(0, 0, 0, 0.28);
  border: 1px solid var(--border);
  border-radius: 10px;
}

.inspect__pos {
  color: var(--text);
  font-weight: 600;
}

.inspect__tag {
  color: var(--text-dim);
}
</style>
