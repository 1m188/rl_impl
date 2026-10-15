<script setup lang="ts">
import { computed } from 'vue';

import { DISPLAY_MODES, type DisplayMode } from '../domain/qTableView';
import { SPEEDS, type SimulationApi } from '../composables/useSimulation';

const props = defineProps<{
  sim: SimulationApi;
  displayMode: DisplayMode;
}>();

const emit = defineEmits<{
  (event: 'update:displayMode', value: DisplayMode): void;
}>();

const ALGORITHM_LABEL: Record<string, string> = {
  qlearning: 'Q-Learning',
  sarsa: 'Sarsa',
};

const algorithmLabel = computed(() =>
  props.sim.algorithm ? ALGORITHM_LABEL[props.sim.algorithm] : '未运行',
);

const runStateLabel = computed(
  () => ({ idle: '空闲', running: '运行中', paused: '已暂停' })[props.sim.runState],
);

/** 最近 20 集的平均步数 —— 比"累计步数"更能说明学得怎么样 */
const recentAverageSteps = computed(() => {
  const window = props.sim.history.slice(-20);
  if (window.length === 0) return null;
  return window.reduce((sum, record) => sum + record.steps, 0) / window.length;
});
</script>

<template>
  <div class="panel">
    <section>
      <h2>算法与运行</h2>
      <div class="row">
        <button type="button" class="primary" @click="sim.start('qlearning')">
          Q-Learning start
        </button>
        <button type="button" class="primary" @click="sim.start('sarsa')">Sarsa start</button>
      </div>
      <div class="row">
        <button type="button" :disabled="!sim.running" @click="sim.pause()">pause</button>
        <button type="button" :disabled="sim.runState !== 'paused'" @click="sim.resume()">
          resume
        </button>
        <button type="button" :disabled="sim.runState !== 'paused'" @click="sim.stepOnce()">
          step
        </button>
        <button type="button" :disabled="sim.runState === 'idle'" @click="sim.stop()">
          stop
        </button>
      </div>
      <div class="row">
        <button type="button" @click="sim.resetBall()">initialize blue ball's position</button>
      </div>
      <p class="hint">
        stop 会丢掉这一轮学到的 Q 表（与原实现一致），想保留请用 pause。
      </p>
    </section>

    <section>
      <h2>速度</h2>
      <div class="row">
        <button
          v-for="value in SPEEDS"
          :key="value"
          type="button"
          class="chip"
          :class="{ active: sim.speed === value }"
          @click="sim.speed = value"
        >
          {{ value }}x
        </button>
      </div>
      <p class="hint">
        1x = 每 50ms 一步（与原实现一致），当前每步
        {{ sim.stepIntervalMs.toFixed(1) }}ms。
      </p>
      <div class="row">
        <button type="button" :disabled="!sim.hasAgent" @click="sim.trainEpisodes(100)">
          快进 100 集
        </button>
        <button type="button" :disabled="!sim.hasAgent" @click="sim.trainEpisodes(1000)">
          快进 1000 集
        </button>
      </div>
      <p class="hint">快进不等渲染，直接跑完指定集数，用来快速看 Q 表收敛。</p>
    </section>

    <section>
      <h2>参数与种子</h2>
      <div class="params">
        <label>
          α 学习率
          <input v-model.number="sim.params.alpha" type="number" min="0" max="1" step="0.05" />
        </label>
        <label>
          γ 折扣率
          <input v-model.number="sim.params.gamma" type="number" min="0" max="1" step="0.05" />
        </label>
        <label>
          ε 探索率
          <input v-model.number="sim.params.epsilon" type="number" min="0" max="1" step="0.01" />
        </label>
        <label class="seed">
          随机种子
          <input v-model="sim.seedText" type="text" inputmode="numeric" placeholder="留空 = 随机" />
        </label>
      </div>
      <p class="hint">
        参数与种子在点 start 时生效。种子用于复现同一段学习轨迹——原程序每次运行的动作顺序都不一样，
        这里固定下来后就完全可复现了。
        <span v-if="sim.seedStatus === 'invalid'" class="warn">种子不是数字，将按随机处理。</span>
        <span v-else-if="sim.appliedSeed !== null">
          本次运行使用种子 {{ sim.appliedSeed }}。
        </span>
      </p>
    </section>

    <section>
      <h2>棋盘叠加</h2>
      <div class="row">
        <button
          v-for="mode in DISPLAY_MODES"
          :key="mode.value"
          type="button"
          class="chip"
          :class="{ active: displayMode === mode.value }"
          @click="emit('update:displayMode', mode.value)"
        >
          {{ mode.label }}
        </button>
      </div>
    </section>

    <section>
      <h2>状态</h2>
      <dl class="status">
        <div>
          <dt>算法</dt>
          <dd data-testid="algorithm">{{ algorithmLabel }}</dd>
        </div>
        <div>
          <dt>运行</dt>
          <dd data-testid="run-state">{{ runStateLabel }}</dd>
        </div>
        <div>
          <dt>已完成回合</dt>
          <dd data-testid="episode">{{ sim.episode }}</dd>
        </div>
        <div>
          <dt>累计步数</dt>
          <dd data-testid="steps">{{ sim.steps }}</dd>
        </div>
        <div>
          <dt>Q 表状态数</dt>
          <dd data-testid="state-count">{{ sim.tableView.knownCount }} / {{ sim.tableView.stateCount }}</dd>
        </div>
        <div>
          <dt>近 20 集平均步数</dt>
          <dd data-testid="recent-steps">
            {{ recentAverageSteps === null ? '—' : recentAverageSteps.toFixed(1) }}
          </dd>
        </div>
      </dl>
    </section>
  </div>
</template>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  gap: 18px;
  width: min(380px, 100%);
}

section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

h2 {
  margin: 0;
  font-size: 13px;
  font-weight: 600;
  color: #333;
}

.row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

button {
  padding: 7px 10px;
  font: inherit;
  font-size: 13px;
  color: #1b1b1b;
  background: #f4f4f4;
  border: 1px solid #c9c9c9;
  border-radius: 6px;
  cursor: pointer;
}

button:hover:not(:disabled) {
  background: #eaeaea;
}

button:disabled {
  color: #9a9a9a;
  cursor: not-allowed;
}

button.primary {
  color: #fff;
  background: #1a4fd6;
  border-color: #1a4fd6;
}

button.primary:hover:not(:disabled) {
  background: #1642b4;
}

button.chip {
  padding: 4px 10px;
  font-size: 12px;
  border-radius: 999px;
}

button.chip.active {
  color: #fff;
  background: #1a4fd6;
  border-color: #1a4fd6;
}

.params {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 10px;
  font-size: 12px;
  color: #4a4a4a;
}

.params label {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.params .seed {
  grid-column: 1 / -1;
}

input {
  padding: 5px 6px;
  font: inherit;
  font-size: 13px;
  color: #1b1b1b;
  background: #fff;
  border: 1px solid #c9c9c9;
  border-radius: 5px;
}

.status {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 12px;
  margin: 0;
  font-size: 12px;
}

.status div {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.status dt {
  color: #6b6b6b;
}

.status dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}

.hint {
  margin: 0;
  font-size: 11px;
  line-height: 1.6;
  color: #7a7a7a;
}

.warn {
  color: #b3261e;
}
</style>
