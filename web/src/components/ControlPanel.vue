<script setup lang="ts">
import { computed } from 'vue';

import { DISPLAY_MODES, type DisplayMode } from '../domain/qTableView';
import { SPEEDS, type SimulationApi } from '../composables/useSimulation';

const props = defineProps<{
  sim: SimulationApi;
  /** 不传就不显示"棋盘叠加"这一段（贪吃蛇没有叠加层） */
  displayMode?: DisplayMode;
  /** 不传就不显示"初始化"按钮（不同游戏文案不同） */
  resetLabel?: string;
  /** Q 表状态数的分母：找红球是固定的格子数，贪吃蛇没有上限 */
  stateTotal?: number;
}>();

const emit = defineEmits<{
  (event: 'update:displayMode', value: DisplayMode): void;
}>();

const ALGORITHM_LABEL: Record<string, string> = {
  qlearning: 'Q-Learning',
  sarsa: 'Sarsa',
};

const RUN_STATE_LABEL: Record<string, string> = {
  idle: '空闲',
  running: '运行中',
  paused: '已暂停',
};

const algorithmLabel = computed(() =>
  props.sim.algorithm ? ALGORITHM_LABEL[props.sim.algorithm] : '未运行',
);

const runStateLabel = computed(() => RUN_STATE_LABEL[props.sim.runState]);

/** 分段控件的滑块位置 */
const speedIndex = computed(() => Math.max(0, SPEEDS.indexOf(props.sim.speed)));

/** 最近 20 集的平均步数 —— 比"累计步数"更能说明学得怎么样 */
const recentAverageSteps = computed(() => {
  const window = props.sim.history.slice(-20);
  if (window.length === 0) return null;
  return window.reduce((sum, record) => sum + record.steps, 0) / window.length;
});

const stats = computed(() => [
  { key: 'episode', label: '已完成回合', value: String(props.sim.episode) },
  { key: 'steps', label: '累计步数', value: String(props.sim.steps) },
  {
    key: 'state-count',
    label: 'Q 表状态数',
    value:
      props.stateTotal === undefined
        ? String(props.sim.knownStates)
        : `${props.sim.knownStates} / ${props.stateTotal}`,
  },
  {
    key: 'recent-steps',
    label: `近 ${Math.min(20, props.sim.history.length) || 20} 集平均步数`,
    value: recentAverageSteps.value === null ? '—' : recentAverageSteps.value.toFixed(1),
  },
]);
</script>

<template>
  <section class="card panel" :style="{ '--i': 1 }">
    <header class="card__head">
      <h2 class="card__title">控制台</h2>
      <span class="pill" :class="`pill--${sim.runState}`">
        <span class="pill__dot" />
        <span data-testid="run-state">{{ runStateLabel }}</span>
      </span>
    </header>

    <section class="group">
      <h3 class="group__title">算法</h3>
      <div class="row">
        <button type="button" class="btn btn--primary" @click="sim.start('qlearning')">
          <span class="btn__icon">▶</span>Q-Learning start
        </button>
        <button type="button" class="btn btn--violet" @click="sim.start('sarsa')">
          <span class="btn__icon">▶</span>Sarsa start
        </button>
      </div>
      <div class="row">
        <button
          type="button"
          class="btn btn--ghost"
          :disabled="!sim.running"
          @click="sim.pause()"
        >
          <span class="btn__icon">❚❚</span>pause
        </button>
        <button
          type="button"
          class="btn btn--ghost"
          :disabled="sim.runState !== 'paused'"
          @click="sim.resume()"
        >
          <span class="btn__icon">▶</span>resume
        </button>
        <button
          type="button"
          class="btn btn--ghost"
          :disabled="sim.runState !== 'paused'"
          @click="sim.stepOnce()"
        >
          <span class="btn__icon">⏭</span>step
        </button>
        <button
          type="button"
          class="btn btn--ghost"
          :disabled="sim.runState === 'idle'"
          @click="sim.stop()"
        >
          <span class="btn__icon">■</span>stop
        </button>
      </div>
      <button v-if="resetLabel" type="button" class="btn btn--ghost wide" @click="sim.resetState()">
        <span class="btn__icon">↺</span>{{ resetLabel }}
      </button>
      <p class="hint">stop 会丢掉这一轮学到的 Q 表（与原实现一致），想保留请用 pause。</p>
    </section>

    <section class="group">
      <h3 class="group__title">速度</h3>
      <div class="seg" :style="{ '--count': SPEEDS.length, '--index': speedIndex }">
        <span class="seg__thumb" />
        <button
          v-for="value in SPEEDS"
          :key="value"
          type="button"
          class="seg__item"
          :class="{ active: sim.speed === value }"
          @click="sim.speed = value"
        >
          {{ value }}x
        </button>
      </div>
      <div class="row">
        <button
          type="button"
          class="btn"
          :disabled="!sim.hasAgent"
          @click="sim.trainEpisodes(100)"
        >
          快进 100 集
        </button>
        <button
          type="button"
          class="btn"
          :disabled="!sim.hasAgent"
          @click="sim.trainEpisodes(1000)"
        >
          快进 1000 集
        </button>
      </div>
      <p class="hint">
        1x = 每 50ms 一步（与原实现一致），当前每步
        <span class="num">{{ sim.stepIntervalMs.toFixed(1) }}</span> ms。
        快进不等渲染，直接跑完指定集数。
      </p>
    </section>

    <section class="group">
      <h3 class="group__title">参数与种子</h3>
      <div class="params">
        <label class="field">
          α 学习率
          <input v-model.number="sim.params.alpha" type="number" min="0" max="1" step="0.05" />
        </label>
        <label class="field">
          γ 折扣率
          <input v-model.number="sim.params.gamma" type="number" min="0" max="1" step="0.05" />
        </label>
        <label class="field">
          ε 探索率
          <input v-model.number="sim.params.epsilon" type="number" min="0" max="1" step="0.01" />
        </label>
        <label class="field seed">
          随机种子
          <input
            v-model="sim.seedText"
            type="text"
            inputmode="numeric"
            placeholder="留空 = 随机"
          />
        </label>
      </div>
      <p class="hint">
        参数与种子在点 start 时生效。种子用于复现同一段学习轨迹 —— 原程序每次运行的动作顺序都不一样，
        固定下来后就完全可复现了。
        <span v-if="sim.seedStatus === 'invalid'" class="warn">种子不是数字，将按随机处理。</span>
        <span v-else-if="sim.appliedSeed !== null" class="ok">
          本次运行使用种子 <span class="num">{{ sim.appliedSeed }}</span
          >。
        </span>
      </p>
    </section>

    <section v-if="displayMode !== undefined" class="group">
      <h3 class="group__title">棋盘叠加</h3>
      <div class="modes">
        <button
          v-for="mode in DISPLAY_MODES"
          :key="mode.value"
          type="button"
          class="mode"
          :class="{ 'mode--active': displayMode === mode.value }"
          @click="emit('update:displayMode', mode.value)"
        >
          {{ mode.label }}
        </button>
      </div>
    </section>

    <section class="group">
      <h3 class="group__title">状态</h3>
      <dl class="stats">
        <div v-for="item in stats" :key="item.key" class="stat">
          <dt>{{ item.label }}</dt>
          <dd class="num" :data-testid="item.key">{{ item.value }}</dd>
        </div>
        <div class="stat">
          <dt>算法</dt>
          <dd class="algo" data-testid="algorithm">{{ algorithmLabel }}</dd>
        </div>
      </dl>
    </section>
  </section>
</template>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.group {
  display: flex;
  flex-direction: column;
  gap: 9px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
}

.group:first-of-type {
  padding-top: 0;
  border-top: 0;
}

.group__title {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
}

.row {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.btn {
  flex: 1 1 auto;
  white-space: nowrap;
}

.btn--ghost {
  flex: 0 1 auto;
}

.wide {
  width: 100%;
}

/* 运行状态胶囊 */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  border-radius: 999px;
  transition:
    color var(--dur) var(--ease-out),
    background var(--dur) var(--ease-out),
    border-color var(--dur) var(--ease-out);
}

.pill__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--text-dim);
  transition: background var(--dur) linear;
}

.pill--running {
  color: var(--good);
  background: rgba(52, 211, 153, 0.12);
  border-color: rgba(52, 211, 153, 0.4);
}

.pill--running .pill__dot {
  background: var(--good);
  animation: dot-pulse 1.4s ease-in-out infinite;
}

.pill--paused {
  color: var(--warn);
  background: rgba(251, 191, 36, 0.12);
  border-color: rgba(251, 191, 36, 0.4);
}

.pill--paused .pill__dot {
  background: var(--warn);
}

@keyframes dot-pulse {
  0%,
  100% {
    box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(52, 211, 153, 0);
  }
}

/* 参数输入 */
.params {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.params .seed {
  grid-column: 1 / -1;
}

/* 叠加模式：2 x 2 的开关块 */
.modes {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 7px;
}

.mode {
  padding: 9px 10px;
  font: inherit;
  font-size: 12px;
  color: var(--text-muted);
  background: rgba(0, 0, 0, 0.26);
  border: 1px solid var(--border);
  border-radius: 10px;
  cursor: pointer;
  transition:
    color var(--dur) var(--ease-out),
    background var(--dur) var(--ease-out),
    border-color var(--dur) var(--ease-out),
    box-shadow var(--dur) var(--ease-out),
    transform var(--dur-fast) var(--ease-out);
}

.mode:hover {
  transform: translateY(-1px);
  color: var(--text);
  border-color: var(--border-strong);
}

.mode--active {
  color: var(--text);
  background: linear-gradient(180deg, rgba(34, 211, 238, 0.22), rgba(34, 211, 238, 0.08));
  border-color: rgba(34, 211, 238, 0.55);
  box-shadow: 0 0 22px -10px rgba(34, 211, 238, 0.9);
}

/* 状态磁贴 */
.stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin: 0;
}

.stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 9px 11px;
  background: rgba(0, 0, 0, 0.26);
  border: 1px solid var(--border);
  border-radius: 11px;
  transition: border-color var(--dur) var(--ease-out);
}

.stat:hover {
  border-color: var(--border-strong);
}

.stat dt {
  font-size: 11px;
  color: var(--text-dim);
}

.stat dd {
  margin: 0;
  font-size: 17px;
  font-weight: 600;
  color: var(--text);
  line-height: 1.2;
}

.stat dd.algo {
  font-size: 15px;
  color: var(--accent);
}

.hint {
  margin: 0;
  font-size: 11px;
  line-height: 1.65;
  color: var(--text-dim);
}

.warn {
  color: var(--bad);
}

.ok {
  color: var(--good);
}
</style>
