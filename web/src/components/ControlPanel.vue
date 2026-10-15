<script setup lang="ts">
import type { AlgorithmKind } from '../composables/useSimulation';

const props = defineProps<{
  running: boolean;
  algorithm: AlgorithmKind | null;
  episode: number;
  steps: number;
  stateCount: number;
  alpha: number;
  gamma: number;
  epsilon: number;
}>();

const emit = defineEmits<{
  (event: 'start', kind: AlgorithmKind): void;
  (event: 'stop'): void;
  (event: 'reset-ball'): void;
}>();

const algorithmLabel: Record<AlgorithmKind, string> = {
  qlearning: 'Q-Learning',
  sarsa: 'Sarsa',
};

function currentLabel(): string {
  return props.algorithm ? algorithmLabel[props.algorithm] : '未运行';
}
</script>

<template>
  <!-- 按钮文字照抄原来右键菜单的四项，功能一一对应 -->
  <div class="panel">
    <div class="actions">
      <button type="button" class="primary" @click="emit('start', 'qlearning')">
        Q-Learning start
      </button>
      <button type="button" class="primary" @click="emit('start', 'sarsa')">Sarsa start</button>
      <button type="button" :disabled="!running" @click="emit('stop')">stop</button>
      <button type="button" @click="emit('reset-ball')">initialize blue ball's position</button>
    </div>

    <dl class="status">
      <div>
        <dt>当前算法</dt>
        <dd data-testid="algorithm">{{ currentLabel() }}</dd>
      </div>
      <div>
        <dt>已完成回合</dt>
        <dd data-testid="episode">{{ episode }}</dd>
      </div>
      <div>
        <dt>累计步数</dt>
        <dd data-testid="steps">{{ steps }}</dd>
      </div>
      <div>
        <dt>Q 表状态数</dt>
        <dd data-testid="state-count">{{ stateCount }} / 16</dd>
      </div>
    </dl>

    <p class="params">
      超参：α = {{ alpha }}，γ = {{ gamma }}，ε = {{ epsilon }}（与原 Python 版一致）
    </p>
  </div>
</template>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 240px;
}

.actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

button {
  padding: 8px 12px;
  font: inherit;
  font-size: 14px;
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

button.primary:hover {
  background: #1642b4;
}

.status {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 12px;
  margin: 0;
  font-size: 13px;
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

.params {
  margin: 0;
  font-size: 12px;
  color: #6b6b6b;
}
</style>
