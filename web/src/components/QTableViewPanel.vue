<script setup lang="ts">
import { computed } from 'vue';

import { ACTION_ORDER } from '../domain/actions';
import type { FindBallEnv } from '../domain/env';
import type { QTableView } from '../domain/qTableView';
import { heatRgba } from '../ui/heat';

const props = defineProps<{
  view: QTableView;
  env: FindBallEnv;
}>();

const rows = computed(() =>
  props.view.cells.map((cell) => ({
    ...cell,
    label: `(${cell.x},${cell.y})`,
    badge: cell.state === props.env.goal ? '终点' : props.env.traps.has(cell.state) ? '陷阱' : '',
  })),
);

const num = (value: number | undefined) => (value === undefined ? '—' : value.toFixed(3));

/** 单元格底色也按数值深浅来，表格本身就成了一张热力图 */
function tint(value: number | undefined): Record<string, string> {
  if (value === undefined) return {};
  return { background: heatRgba(value, props.view.min, props.view.max, 0.04, 0.3) };
}

const rangeLabel = computed(() => {
  if (props.view.knownCount === 0) return '还没有数据';
  return `取值 ${props.view.min.toFixed(2)} ~ ${props.view.max.toFixed(2)}`;
});
</script>

<template>
  <details class="card qtable" :style="{ '--i': 4 }">
    <summary>
      <span class="card__title">Q 表原始数值</span>
      <span class="summary__meta num">
        {{ view.knownCount }} / {{ view.stateCount }} 个状态 · {{ rangeLabel }}
      </span>
      <span class="chev">›</span>
    </summary>

    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th scope="col">状态</th>
            <th v-for="action in ACTION_ORDER" :key="action" scope="col">{{ action }}</th>
            <th scope="col">最优</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.state" :class="{ 'row--terminal': row.terminal }">
            <th scope="row">
              {{ row.label }}
              <span v-if="row.badge" class="badge">{{ row.badge }}</span>
            </th>
            <td
              v-for="action in ACTION_ORDER"
              :key="action"
              :style="tint(row.values[action])"
              :class="{ best: row.policy === action && !row.terminal }"
            >
              {{ num(row.values[action]) }}
            </td>
            <td class="policy">{{ row.terminal || !row.policy ? '—' : row.policy }}</td>
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

.scroll {
  max-height: 320px;
  overflow: auto;
  margin-top: 14px;
  border-radius: 10px;
}

table {
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
  width: 100%;
}

th,
td {
  padding: 4px 9px;
  text-align: right;
  white-space: nowrap;
  transition: background var(--dur) var(--ease-out);
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
}

td.best {
  font-weight: 700;
  color: var(--accent);
  background: var(--accent-soft);
  box-shadow: inset 0 0 0 1px rgba(34, 211, 238, 0.35);
}

td.policy {
  color: var(--violet);
}

.row--terminal td,
.row--terminal th {
  color: var(--text-dim);
}

.badge {
  margin-left: 5px;
  padding: 0 5px;
  font-size: 10px;
  font-family: var(--font);
  color: var(--text-dim);
  background: rgba(255, 255, 255, 0.07);
  border-radius: 4px;
}
</style>
