<script setup lang="ts">
import { computed } from 'vue';

import type { StepResult } from '../domain/agent';
import type { FindBallEnv } from '../domain/env';
import type { AlgorithmKind } from '../composables/useSimulation';

const props = defineProps<{
  transition: StepResult | null;
  kind: AlgorithmKind | null;
  env: FindBallEnv;
  alpha: number;
  gamma: number;
}>();

const ACTION_LABEL: Record<string, string> = {
  UP: 'UP ↑',
  DOWN: 'DOWN ↓',
  LEFT: 'LEFT ←',
  RIGHT: 'RIGHT →',
};

const target = computed(() =>
  props.kind === 'sarsa' ? "r + γ·Q(s',a')" : "r + γ·maxQ(s')",
);

const at = (state: number) => `(${props.env.x(state)}, ${props.env.y(state)})`;
const num = (value: number) => value.toFixed(3);
</script>

<template>
  <section class="step">
    <h2>最近一步</h2>

    <p v-if="!transition" class="step__empty">还没有走过一步。</p>

    <template v-else>
      <ol class="chain">
        <li><span class="key">s</span>{{ at(transition.state) }}</li>
        <li><span class="key">a</span>{{ ACTION_LABEL[transition.action] }}</li>
        <li><span class="key">r</span>{{ num(transition.reward) }}</li>
        <li><span class="key">s'</span>{{ at(transition.next) }}</li>
      </ol>

      <p class="formula">
        Q(s,a) ← (1−{{ alpha }})·Q(s,a) + {{ alpha }}·[ r + {{ gamma }}·… ]，
        这里的目标值 {{ target }} = <strong>{{ num(transition.tdTarget) }}</strong>
      </p>

      <p class="update">
        Q({{ at(transition.state) }}, {{ transition.action }})：
        <span class="before">{{ num(transition.qBefore) }}</span>
        →
        <span class="after">{{ num(transition.qAfter) }}</span>
      </p>
    </template>
  </section>
</template>

<style scoped>
.step {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  background: #fff;
  border: 1px solid #e2e2e2;
  border-radius: 10px;
}

h2 {
  margin: 0;
  font-size: 15px;
}

.step__empty,
.formula,
.update {
  margin: 0;
  font-size: 12px;
  color: #4a4a4a;
  line-height: 1.7;
}

.chain {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.key {
  display: inline-block;
  min-width: 16px;
  margin-right: 4px;
  padding: 0 4px;
  color: #6b6b6b;
  background: #f0f0f0;
  border-radius: 4px;
  font-size: 11px;
}

.update {
  font-variant-numeric: tabular-nums;
}

.before {
  color: #8a8a8a;
}

.after {
  font-weight: 700;
  color: #1a4fd6;
}
</style>
