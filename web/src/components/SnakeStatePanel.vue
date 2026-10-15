<script setup lang="ts">
import { computed } from 'vue';

import { ACTION_ORDER, type Action } from '../domain/actions';
import { NEIGHBOR_OFFSETS, type SnakeFeatures, type SnakeSnapshot } from '../domain/snake';

const props = defineProps<{
  features: SnakeFeatures;
  snapshot: SnakeSnapshot;
  values: Partial<Record<Action, number>> | null;
}>();

const ACTION_LABEL: Record<Action, string> = { UP: '上 ↑', DOWN: '下 ↓', LEFT: '左 ←', RIGHT: '右 →' };

/**
 * "AI 视角"：算法实际看到的就是这 9 个格子 ——
 * 中间是蛇头，周围 8 格标出危险位，再叠上食物相对蛇头的方位。
 */
const slots = computed(() => {
  const grid = Array.from({ length: 9 }, (_, index) => ({
    index,
    danger: false,
    food: false,
    head: false,
  }));

  NEIGHBOR_OFFSETS.forEach(([dx, dy], i) => {
    const slot = grid[(dy + 1) * 3 + (dx + 1)];
    slot.danger = props.features.dangerous[i];
  });

  const foodCol = props.features.food.dx + 1;
  const foodRow = props.features.food.dy + 1;
  if (!(foodCol === 1 && foodRow === 1)) grid[foodRow * 3 + foodCol].food = true;
  grid[4].head = true;

  return grid;
});

/** 四个动作当前的价值（Q 表里这个状态那一行） */
const rows = computed(() => {
  const values = props.values;
  if (!values) return null;
  const list = ACTION_ORDER.map((action) => ({ action, value: values[action] }));
  const known = list.filter((item) => item.value !== undefined) as Array<{
    action: Action;
    value: number;
  }>;
  if (known.length === 0) return null;
  const best = Math.max(...known.map((item) => item.value));
  const worst = Math.min(...known.map((item) => item.value));
  const span = best - worst || 1;
  return known.map((item) => ({
    ...item,
    ratio: (item.value - worst) / span,
    best: item.value === best,
  }));
});

const directionLabel: Record<string, string> = { UP: '上', DOWN: '下', LEFT: '左', RIGHT: '右' };
</script>

<template>
  <section class="card state-panel" :style="{ '--i': 2 }">
    <header class="card__head">
      <h2 class="card__title">AI 视角</h2>
      <span class="key num">状态键 {{ features.terminal ? '终止' : features.key }}</span>
    </header>

    <div class="view">
      <div class="mini" aria-hidden="true">
        <span
          v-for="slot in slots"
          :key="slot.index"
          class="mini__cell"
          :class="{
            'mini__cell--danger': slot.danger,
            'mini__cell--head': slot.head,
            'mini__cell--food': slot.food,
          }"
        >
          <span v-if="slot.head" class="mini__head" />
          <span v-if="slot.food" class="mini__food" />
        </span>
      </div>

      <dl class="facts">
        <div>
          <dt>朝向</dt>
          <dd>{{ directionLabel[snapshot.direction] }}</dd>
        </div>
        <div>
          <dt>食物方位</dt>
          <dd>
            {{ features.food.dy < 0 ? '上' : features.food.dy > 0 ? '下' : '' }}{{
              features.food.dx < 0 ? '左' : features.food.dx > 0 ? '右' : ''
            }}{{ features.food.dx === 0 && features.food.dy === 0 ? '—' : '' }}
          </dd>
        </div>
        <div>
          <dt>长度</dt>
          <dd class="num">{{ snapshot.body.length }}</dd>
        </div>
        <div>
          <dt>得分</dt>
          <dd class="num">{{ snapshot.score }}</dd>
        </div>
      </dl>
    </div>

    <p v-if="!rows" class="empty">这个状态还没进过 Q 表。</p>

    <ul v-else class="values">
      <li v-for="row in rows" :key="row.action" :class="{ 'values--best': row.best }">
        <span class="values__label">{{ ACTION_LABEL[row.action] }}</span>
        <span class="values__bar"><i :style="{ width: `${Math.max(4, row.ratio * 100)}%` }" /></span>
        <span class="values__num num">{{ row.value.toFixed(2) }}</span>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.state-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.key {
  font-size: 11px;
  color: var(--text-dim);
}

.view {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

/* 3x3 的"蛇头视野" */
.mini {
  display: grid;
  grid-template-columns: repeat(3, 30px);
  grid-template-rows: repeat(3, 30px);
  gap: 3px;
  padding: 6px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--border);
  border-radius: 12px;
}

.mini__cell {
  position: relative;
  display: grid;
  place-items: center;
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.05);
  transition: background var(--dur) var(--ease-out);
}

.mini__cell--danger {
  background: rgba(251, 113, 133, 0.42);
  box-shadow: inset 0 0 0 1px rgba(251, 113, 133, 0.6);
}

.mini__cell--head {
  background: linear-gradient(180deg, #7aa8ff, #1d4ed8);
  box-shadow: 0 0 14px -2px rgba(37, 99, 235, 0.9);
}

.mini__head {
  width: 12px;
  height: 12px;
  border-radius: 4px;
  background: #eaf2ff;
  opacity: 0.9;
}

.mini__cell--food {
  background: rgba(251, 191, 36, 0.28);
}

.mini__food {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #ff9a9a, #d81f1f);
  animation: mini-food 1.6s ease-in-out infinite;
}

@keyframes mini-food {
  0%,
  100% {
    box-shadow: 0 0 6px rgba(216, 31, 31, 0.5);
  }
  50% {
    box-shadow: 0 0 14px rgba(216, 31, 31, 0.95);
  }
}

.facts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 16px;
  margin: 0;
  font-size: 12px;
}

.facts div {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.facts dt {
  color: var(--text-dim);
  font-size: 11px;
}

.facts dd {
  margin: 0;
  font-weight: 600;
}

.empty {
  margin: 0;
  font-size: 12px;
  color: var(--text-dim);
}

.values {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: 12px;
}

.values li {
  display: grid;
  grid-template-columns: 46px 1fr 56px;
  align-items: center;
  gap: 8px;
  padding: 3px 6px;
  border-radius: 8px;
  transition: background var(--dur) var(--ease-out);
}

.values--best {
  background: var(--accent-soft);
}

.values__label {
  color: var(--text-muted);
}

.values--best .values__label {
  color: var(--accent);
  font-weight: 600;
}

.values__bar {
  height: 6px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.35);
  overflow: hidden;
}

.values__bar i {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(148, 163, 184, 0.8), var(--accent));
  transition: width var(--dur) var(--ease-out);
}

.values--best .values__bar i {
  background: linear-gradient(90deg, rgba(34, 211, 238, 0.6), #7ff0ff);
}

.values__num {
  text-align: right;
  color: var(--text-muted);
}

.values--best .values__num {
  color: var(--text);
  font-weight: 700;
}
</style>
