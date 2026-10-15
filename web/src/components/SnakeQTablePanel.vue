<script setup lang="ts">
import { computed, ref } from 'vue';

import { ACTION_ORDER, type Action } from '../domain/actions';
import { NEIGHBOR_OFFSETS, type SnakeEnv, type SnakeFeatures } from '../domain/snake';

const props = defineProps<{
  env: SnakeEnv;
  /** 传进来自动跟踪 Q 表变化（父组件用它触发重算） */
  revision: number;
  agentTable: Map<number, Map<Action, number>> | null;
}>();

/** 折叠着的时候不算，省得每 tick 都排一遍几千个状态 */
const open = ref(false);

const DIRECTION_LABEL: Record<Action, string> = { UP: '上', DOWN: '下', LEFT: '左', RIGHT: '右' };
const NEIGHBOR_LABEL = ['左上', '上', '右上', '右', '右下', '下', '左下', '左'];

function describe(features: SnakeFeatures): string {
  const danger = NEIGHBOR_OFFSETS.map((_, i) => (features.dangerous[i] ? NEIGHBOR_LABEL[i] : null)).filter(
    (item): item is string => item !== null,
  );
  const vertical = features.food.dy < 0 ? '上' : features.food.dy > 0 ? '下' : '';
  const horizontal = features.food.dx < 0 ? '左' : features.food.dx > 0 ? '右' : '';
  const food = `${vertical}${horizontal}` || '正前';
  const blocked = danger.length > 0 ? danger.join('') : '无';
  return `危险:${blocked} · 食物:${food} · 朝:${DIRECTION_LABEL[features.direction]}`;
}

const rows = computed(() => {
  ref(props.revision);
  if (!open.value || !props.agentTable) return [];

  const entries: Array<{
    state: number;
    label: string;
    values: Array<{ action: Action; value: number | undefined }>;
    policy: Action | null;
    best: number;
  }> = [];

  for (const [state, row] of props.agentTable) {
    let best = Number.NEGATIVE_INFINITY;
    let policy: Action | null = null;
    for (const action of ACTION_ORDER) {
      const value = row.get(action);
      if (value === undefined) continue;
      if (value > best) {
        best = value;
        policy = action;
      }
    }
    entries.push({
      state,
      label: describe(props.env.features(state)),
      values: ACTION_ORDER.map((action) => ({ action, value: row.get(action) })),
      policy,
      best,
    });
  }

  entries.sort((a, b) => b.best - a.best);
  return entries.slice(0, 40);
});

const total = computed(() => {
  ref(props.revision);
  return props.agentTable?.size ?? 0;
});

const num = (value: number | undefined) => (value === undefined ? '—' : value.toFixed(2));
</script>

<template>
  <details class="card qtable" :style="{ '--i': 4 }" @toggle="open = ($event.target as HTMLDetailsElement).open">
    <summary>
      <span class="card__title">Q 表</span>
      <span class="summary__meta num">{{ total }} 个状态</span>
      <span class="chev">›</span>
    </summary>

    <p class="hint">按最优动作的价值从高到低列出前 40 行。</p>

    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">状态（蛇头局部局势）</th>
            <th v-for="action in ACTION_ORDER" :key="action" scope="col">{{ action }}</th>
            <th scope="col">最优</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.state">
            <th scope="row">{{ row.label }}</th>
            <td
              v-for="item in row.values"
              :key="item.action"
              :class="{ best: row.policy === item.action }"
            >
              {{ num(item.value) }}
            </td>
            <td class="policy">{{ row.policy ?? '—' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </details>
</template>

<style scoped>
.qtable {
  font-size: 12px;
}

summary {
  display: flex;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  list-style: none;
}

summary::-webkit-details-marker {
  display: none;
}

.summary__meta {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-dim);
}

.chev {
  display: inline-block;
  color: var(--text-dim);
  font-size: 16px;
  transform: rotate(90deg);
  transition: transform var(--dur) var(--ease-out);
}

.qtable[open] .chev {
  transform: rotate(-90deg);
}

.hint {
  margin: 12px 0 0;
  font-size: 11px;
  color: var(--text-dim);
}

.scroll {
  max-height: 320px;
  overflow: auto;
  margin-top: 10px;
  border-radius: 10px;
}

table {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  width: 100%;
}

th,
td {
  padding: 4px 8px;
  text-align: right;
  white-space: nowrap;
}

thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #10151d;
  color: var(--text-dim);
  font-weight: 500;
  border-bottom: 1px solid var(--border);
}

tbody tr {
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

tbody tr:hover {
  background: rgba(255, 255, 255, 0.03);
}

tbody th {
  text-align: left;
  font-weight: 500;
  color: var(--text-muted);
  font-size: 11px;
}

td {
  font-family: var(--font-mono);
  color: var(--text-muted);
}

td.best {
  font-weight: 700;
  color: var(--accent);
  background: var(--accent-soft);
}

td.policy {
  color: var(--violet);
}
</style>
