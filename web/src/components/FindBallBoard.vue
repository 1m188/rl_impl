<script setup lang="ts">
import { computed, ref } from 'vue';

import { ACTION_ORDER, type Action } from '../domain/actions';
import type { FindBallEnv, State } from '../domain/env';
import {
  showsHeat,
  showsPolicy,
  type DisplayMode,
  type QTableView,
  type StateView,
} from '../domain/qTableView';
import { heatRgba } from '../ui/heat';

const props = defineProps<{
  env: FindBallEnv;
  state: State;
  view: QTableView;
  displayMode: DisplayMode;
  trail: State[];
  burst: { key: number; state: State; reason: string } | null;
  moveDurationMs: number;
  teleported: boolean;
  running: boolean;
}>();

const ARROWS: Record<Action, string> = { UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→' };
const ACTION_LABEL: Record<Action, string> = { UP: '上', DOWN: '下', LEFT: '左', RIGHT: '右' };

const hovered = ref<State | null>(null);

function heatColor(cell: StateView): string | undefined {
  if (!showsHeat(props.displayMode)) return undefined;
  if (!cell.known || cell.terminal || cell.best === null) return undefined;
  return heatRgba(cell.best, props.view.min, props.view.max, 0.07, 0.52);
}

interface Cell extends StateView {
  trap: boolean;
  goal: boolean;
  lastCol: boolean;
  lastRow: boolean;
  arrow: string;
  heat: string | undefined;
  burst: { key: number; reason: string } | null;
}

const cells = computed<Cell[]>(() =>
  props.view.cells.map((cell) => {
    const hit = props.burst && props.burst.state === cell.state ? props.burst : null;
    return {
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
      burst: hit ? { key: hit.key, reason: hit.reason } : null,
    };
  }),
);

/** 球的位移用 transform（走 GPU 合成，且被打断时能从当前位置续上） */
const stateTransform = computed(() => ({
  transform: `translate(${props.env.x(props.state) * 100}%, ${props.env.y(props.state) * 100}%)`,
}));

function trailTransform(state: State, index: number) {
  const ratio = 1 - index / (props.trail.length + 1);
  return {
    transform: `translate(${props.env.x(state) * 100}%, ${props.env.y(state) * 100}%) scale(${0.3 + ratio * 0.4})`,
    opacity: String(0.05 + ratio * 0.22),
  };
}

const hoveredCell = computed<StateView | null>(
  () => props.view.cells.find((cell) => cell.state === hovered.value) ?? null,
);

const fmt = (value: number | undefined) => (value === undefined ? '—' : value.toFixed(2));
</script>

<template>
  <section class="card board-card" :style="{ '--i': 0 }">
    <header class="card__head">
      <h2 class="card__title">棋盘</h2>
      <ul class="legend">
        <li><span class="swatch swatch--ball" />蓝球</li>
        <li><span class="swatch swatch--goal" />终点 +10</li>
        <li><span class="swatch swatch--trap" />陷阱 −10</li>
        <li><span class="swatch swatch--heat" />底色 = max Q</li>
      </ul>
    </header>

    <div
      class="board-wrap"
      :style="{
        '--cols': env.width,
        '--rows': env.height,
        '--ball-duration': teleported ? '0ms' : `${moveDurationMs}ms`,
      }"
    >
      <div
        class="board"
        role="img"
        :aria-label="`${env.width} 乘 ${env.height} 的网格，蓝球在第 ${env.x(state) + 1} 列第 ${env.y(state) + 1} 行`"
        @mouseleave="hovered = null"
      >
        <div
          v-for="cell in cells"
          :key="cell.state"
          class="cell"
          :class="{
            'cell--trap': cell.trap,
            'cell--last-col': cell.lastCol,
            'cell--last-row': cell.lastRow,
            'cell--hover': hovered === cell.state,
          }"
          :style="{ background: cell.heat }"
          @mouseenter="hovered = cell.state"
        >
          <!-- 陷阱方块本身也参与命中动画，所以用 key 让它每次重新创建 -->
          <span
            v-if="cell.trap"
            :key="cell.burst?.reason === 'trap' ? `trap-${cell.burst.key}` : 'trap'"
            class="trap"
            :class="{ 'trap--hit': cell.burst?.reason === 'trap' }"
          />

          <span
            v-if="cell.goal"
            :key="cell.burst?.reason === 'goal' ? `goal-${cell.burst.key}` : 'goal'"
            class="goal"
            :class="{ 'goal--hit': cell.burst?.reason === 'goal' }"
          />
          <span
            v-if="cell.burst?.reason === 'goal'"
            :key="`ring-${cell.burst.key}`"
            class="ring"
          />

          <Transition name="arrow">
            <span v-if="cell.arrow" :key="cell.arrow" class="arrow">{{ cell.arrow }}</span>
          </Transition>
        </div>

        <span
          v-for="(trailState, index) in trail"
          :key="index"
          class="trail"
          :style="trailTransform(trailState, index)"
        />

        <div class="agent" :class="{ 'agent--running': running }" :data-cell="`${env.x(state)},${env.y(state)}`" :style="stateTransform">
          <span class="agent__dot" />
        </div>
      </div>

      <div class="inspect">
        <template v-if="hoveredCell">
          <span class="inspect__pos num">({{ hoveredCell.x }}, {{ hoveredCell.y }})</span>
          <span v-if="hoveredCell.terminal" class="inspect__tag">
            {{ env.goal === hoveredCell.state ? '终点' : '陷阱' }}：一集到这里就结束，Q 值恒为 0
          </span>
          <span v-else-if="hoveredCell.known" class="inspect__values">
            <span
              v-for="action in ACTION_ORDER"
              :key="action"
              class="q num"
              :class="{ 'q--best': hoveredCell.policy === action }"
            >
              {{ ACTION_LABEL[action] }} {{ fmt(hoveredCell.values[action]) }}
            </span>
          </span>
          <span v-else class="inspect__tag">这个格子还没被访问过</span>
        </template>
        <span v-else class="inspect__tag">把鼠标移到格子上，可以看它的四个动作价值</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.board-card {
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
  border-radius: 50%;
  display: inline-block;
}

.swatch--ball {
  background: #2563eb;
  box-shadow: 0 0 8px rgba(37, 99, 235, 0.9);
}

.swatch--goal {
  background: #e02020;
  box-shadow: 0 0 8px rgba(224, 32, 32, 0.8);
}

.swatch--trap {
  background: #0b0b0b;
  border-radius: 2px;
}

.swatch--heat {
  border-radius: 2px;
  background: linear-gradient(90deg, rgba(148, 163, 184, 0.35), rgba(249, 115, 22, 0.75));
}

.board-wrap {
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: center;
}

/* 棋盘保持白底：陷阱必须是黑块，反色就没法看了。
   周围用暗色卡片托住，白板反而成了视觉中心。 */
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
    0 30px 60px -30px rgba(34, 211, 238, 0.55),
    0 10px 30px -20px rgba(0, 0, 0, 0.9);
}

.cell {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color var(--dur-slow) var(--ease-out);
}

/* 只画内部网格线，与原 paintEvent 里的循环一致 */
.cell:not(.cell--last-col) {
  border-right: 1px solid rgba(0, 0, 0, 0.75);
}

.cell:not(.cell--last-row) {
  border-bottom: 1px solid rgba(0, 0, 0, 0.75);
}

.cell--hover::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(34, 211, 238, 0.22);
  box-shadow: inset 0 0 0 2px rgba(34, 211, 238, 0.85);
  pointer-events: none;
}

/* 黑色陷阱方块：原实现是每格向内缩进 5px 的矩形 */
.trap {
  position: absolute;
  inset: 5px;
  background: #000;
  border-radius: 3px;
}

.trap--hit {
  animation:
    trap-shake 380ms ease-in-out both,
    trap-flash 620ms var(--ease-out) both;
}

@keyframes trap-shake {
  0%,
  100% {
    transform: translate3d(0, 0, 0);
  }
  20% {
    transform: translate3d(-4px, 1px, 0);
  }
  45% {
    transform: translate3d(4px, -1px, 0);
  }
  70% {
    transform: translate3d(-2px, 0, 0);
  }
}

@keyframes trap-flash {
  0% {
    background: var(--bad);
    box-shadow: 0 0 22px 4px rgba(251, 113, 133, 0.85);
  }
  100% {
    background: #000;
    box-shadow: none;
  }
}

/* 红色终点：向内缩进 5px 的正圆 */
.goal {
  position: absolute;
  inset: 5px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #ff6a6a, #d81f1f 70%);
}

.goal--hit {
  animation: goal-pop 520ms var(--ease-spring) both;
}

@keyframes goal-pop {
  0% {
    transform: scale(1);
  }
  30% {
    transform: scale(1.22);
  }
  100% {
    transform: scale(1);
  }
}

/* 到达终点时向外扩散的一圈光环 */
.ring {
  position: absolute;
  inset: 12px;
  border-radius: 50%;
  border: 2px solid var(--accent);
  animation: ring-out 620ms var(--ease-out) both;
  pointer-events: none;
}

@keyframes ring-out {
  0% {
    opacity: 0.9;
    transform: scale(0.5);
  }
  100% {
    opacity: 0;
    transform: scale(1.35);
  }
}

/* 策略箭头 */
.arrow {
  position: relative;
  z-index: 1;
  font-size: 24px;
  line-height: 1;
  color: rgba(15, 23, 42, 0.82);
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.75);
  pointer-events: none;
}

.arrow-enter-active,
.arrow-leave-active {
  transition:
    opacity var(--dur) var(--ease-out),
    transform var(--dur) var(--ease-spring);
}

.arrow-enter-from,
.arrow-leave-to {
  opacity: 0;
  transform: scale(0.5);
}

.arrow-leave-active {
  position: absolute;
}

/* 轨迹残影 */
.trail {
  position: absolute;
  top: 0;
  left: 0;
  width: calc(100% / var(--cols));
  height: calc(100% / var(--rows));
  pointer-events: none;
  transition:
    transform var(--ball-duration) var(--ease-out),
    opacity var(--dur-slow) linear;
}

.trail::after {
  content: '';
  position: absolute;
  inset: 32%;
  border-radius: 50%;
  background: #2563eb;
  filter: blur(1px);
}

/* 蓝球 */
.agent {
  position: absolute;
  top: 0;
  left: 0;
  width: calc(100% / var(--cols));
  height: calc(100% / var(--rows));
  pointer-events: none;
  z-index: 2;
  transition: transform var(--ball-duration) var(--ease-out);
}

.agent__dot {
  position: absolute;
  inset: 5px;
  border-radius: 50%;
  background: radial-gradient(circle at 34% 28%, #7aa8ff, #1d4ed8 62%, #123a9e);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.35),
    0 6px 14px -6px rgba(29, 78, 216, 0.9);
}

.agent--running .agent__dot {
  animation: ball-pulse 1.5s ease-in-out infinite;
}

@keyframes ball-pulse {
  0%,
  100% {
    box-shadow:
      0 0 0 1px rgba(255, 255, 255, 0.35),
      0 6px 14px -6px rgba(29, 78, 216, 0.9);
  }
  50% {
    box-shadow:
      0 0 0 3px rgba(37, 99, 235, 0.25),
      0 8px 22px -6px rgba(37, 99, 235, 1);
  }
}

/* 鼠标悬停时的读数条：放在棋盘下方，不会被裁切 */
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
  transition: border-color var(--dur) var(--ease-out);
}

.inspect__pos {
  color: var(--text);
  font-weight: 600;
}

.inspect__values {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 8px;
}

.q {
  padding: 2px 7px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid transparent;
  transition:
    background var(--dur) var(--ease-out),
    color var(--dur) var(--ease-out),
    border-color var(--dur) var(--ease-out);
}

.q--best {
  color: #04121a;
  font-weight: 700;
  background: var(--accent);
  border-color: rgba(34, 211, 238, 0.9);
  box-shadow: 0 0 16px -4px rgba(34, 211, 238, 0.9);
}

.inspect__tag {
  color: var(--text-dim);
}
</style>
