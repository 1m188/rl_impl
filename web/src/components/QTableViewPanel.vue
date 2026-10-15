<script setup lang="ts">
import { computed } from 'vue';

import { ACTION_ORDER } from '../domain/actions';
import type { FindBallEnv } from '../domain/env';
import type { QTableView } from '../domain/qTableView';

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
</script>

<template>
  <details class="qtable">
    <summary>
      Q 表原始数值（{{ view.knownCount }} / {{ view.stateCount }} 个状态已知，
      取值 {{ view.knownCount ? `${view.min.toFixed(2)} ~ ${view.max.toFixed(2)}` : '—' }}）
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
  padding: 12px 16px;
  background: #fff;
  border: 1px solid #e2e2e2;
  border-radius: 10px;
  font-size: 12px;
}

summary {
  cursor: pointer;
  color: #4a4a4a;
}

.scroll {
  max-height: 320px;
  overflow: auto;
  margin-top: 10px;
}

table {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
  width: 100%;
}

th,
td {
  padding: 3px 8px;
  text-align: right;
  border-bottom: 1px solid #f0f0f0;
  white-space: nowrap;
}

thead th {
  position: sticky;
  top: 0;
  background: #fff;
  color: #6b6b6b;
  font-weight: 500;
}

tbody th {
  text-align: left;
  font-weight: 500;
  color: #333;
}

td.best {
  font-weight: 700;
  color: #1a4fd6;
  background: #eef3ff;
}

td.policy {
  color: #6b4a00;
}

.row--terminal td {
  color: #b0b0b0;
}

.badge {
  margin-left: 4px;
  padding: 0 4px;
  font-size: 10px;
  color: #fff;
  background: #9a9a9a;
  border-radius: 3px;
}
</style>
