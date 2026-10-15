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

const target = computed(() => (props.kind === 'sarsa' ? "r + γ·Q(s',a')" : "r + γ·maxQ(s')"));

const at = (state: number) => `(${props.env.x(state)}, ${props.env.y(state)})`;
const num = (value: number) => value.toFixed(3);

/** 收益条：宽度按 |r| / 10 计，正负用颜色区分 */
const rewardBar = computed(() => {
  const reward = props.transition?.reward ?? 0;
  const ratio = Math.min(1, Math.abs(reward) / 10);
  return {
    width: `${(ratio * 100).toFixed(1)}%`,
    background: reward >= 0 ? 'var(--good)' : 'var(--bad)',
  };
});

const gained = computed(() => {
  const item = props.transition;
  if (!item) return false;
  return item.qAfter > item.qBefore;
});
</script>

<template>
  <section class="card step" :style="{ '--i': 3 }">
    <header class="card__head">
      <h2 class="card__title">最近一步</h2>
      <div v-if="kind" class="chips">
        <span class="chip">α {{ alpha }}</span>
        <span class="chip">γ {{ gamma }}</span>
        <span class="chip chip--accent">{{ kind === 'sarsa' ? 'on-policy' : 'off-policy' }}</span>
      </div>
    </header>

    <p v-if="!transition" class="empty">还没有走过一步。</p>

    <template v-else>
      <ol class="chain">
        <li><span class="key">s</span>{{ at(transition.state) }}</li>
        <li><span class="key">a</span>{{ ACTION_LABEL[transition.action] }}</li>
        <li><span class="key">r</span>{{ num(transition.reward) }}</li>
        <li><span class="key">s'</span>{{ at(transition.next) }}</li>
      </ol>

      <div class="bar" :title="`本步收益 ${num(transition.reward)}`">
        <span class="bar__fill" :style="rewardBar" />
      </div>

      <p class="formula">
        Q(s,a) ← (1−{{ alpha }})·Q(s,a) + {{ alpha }}·[ r + {{ gamma }}·… ]，目标值
        {{ target }} = <strong class="num">{{ num(transition.tdTarget) }}</strong>
      </p>

      <p class="update">
        Q({{ at(transition.state) }}, {{ transition.action }})：
        <span class="before num">{{ num(transition.qBefore) }}</span>
        <span class="arrow-inline">→</span>
        <span class="after num" :class="{ 'after--up': gained }">
          {{ num(transition.qAfter) }}
        </span>
      </p>
    </template>
  </section>
</template>

<style scoped>
.step {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.empty {
  margin: 0;
  font-size: 12px;
  color: var(--text-dim);
}

.chips {
  display: flex;
  gap: 6px;
}

.chip {
  padding: 2px 8px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border);
  border-radius: 999px;
}

.chip--accent {
  color: var(--accent);
  background: var(--accent-soft);
  border-color: rgba(34, 211, 238, 0.4);
}

/* s → a → r → s' 链条 */
.chain {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 7px;
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.chain li {
  position: relative;
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 9px;
  background: rgba(0, 0, 0, 0.28);
  border: 1px solid var(--border);
  border-radius: 9px;
  transition: border-color var(--dur) var(--ease-out);
}

.chain li:hover {
  border-color: var(--border-strong);
}

/* 箭头用伪元素画：不进 DOM 文本，读起来就是干净的 4 条事实 */
.chain li:not(:last-child)::after {
  content: '→';
  position: absolute;
  right: -13px;
  color: var(--text-dim);
  font-size: 12px;
}

.chain li + li {
  margin-left: 12px;
}

.key {
  display: inline-block;
  min-width: 16px;
  padding: 0 4px;
  text-align: center;
  color: var(--accent);
  background: var(--accent-soft);
  border-radius: 4px;
  font-size: 11px;
  font-family: var(--font-mono);
}

/* 收益条 */
.bar {
  position: relative;
  height: 6px;
  overflow: hidden;
  background: rgba(0, 0, 0, 0.35);
  border-radius: 999px;
}

.bar__fill {
  display: block;
  height: 100%;
  border-radius: 999px;
  transition:
    width var(--dur) var(--ease-out),
    background var(--dur) linear;
}

.formula,
.update {
  margin: 0;
  font-size: 12px;
  line-height: 1.7;
  color: var(--text-muted);
}

.formula strong {
  color: var(--text);
}

.update {
  font-variant-numeric: tabular-nums;
}

.before {
  color: var(--text-dim);
}

.arrow-inline {
  margin: 0 6px;
  color: var(--text-dim);
}

.after {
  font-weight: 700;
  color: var(--accent);
  transition: color var(--dur) var(--ease-out);
}

.after--up {
  color: var(--good);
}
</style>
