<script setup lang="ts">
import { computed, ref } from 'vue';

import FindBallTab from './components/FindBallTab.vue';
import SnakeTab from './components/SnakeTab.vue';
import { useSimulation, type SimulationParams } from './composables/useSimulation';
import { createFindBallEnv } from './domain/env';
import { createSnakeEnv } from './domain/snake';

type TabKey = 'findball' | 'snake';

const TABS: TabKey[] = ['findball', 'snake'];
const TAB_LABEL: Record<TabKey, string> = { findball: '找红球', snake: '贪吃蛇' };

/**
 * 贪吃蛇的状态空间比找红球大得多（约 9216 个压缩状态），
 * 默认探索率要比找红球的 0.01 高，否则大部分状态根本走不到。
 */
const SNAKE_PARAMS: SimulationParams = { alpha: 0.5, gamma: 0.9, epsilon: 0.1 };
/** 坏策略下蛇可能一直绕圈，给一集一个步数上限 */
const SNAKE_MAX_STEPS = 300;

const tab = ref<TabKey>('findball');
const tabIndex = computed(() => Math.max(0, TABS.indexOf(tab.value)));

const findBallEnv = createFindBallEnv();
const snakeEnv = createSnakeEnv();

/** 两个页签各跑各的；切走的那个停掉动画帧但保留状态 */
const findBall = useSimulation(findBallEnv, {
  active: computed(() => tab.value === 'findball'),
});
const snake = useSimulation(snakeEnv, {
  maxStepsPerEpisode: SNAKE_MAX_STEPS,
  defaultParams: SNAKE_PARAMS,
  active: computed(() => tab.value === 'snake'),
});

const activeSim = computed(() => (tab.value === 'findball' ? findBall : snake));

const RUN_STATE_LABEL = {
  idle: '空闲',
  running: '运行中',
  paused: '已暂停',
} as const;

const statusLabel = computed(() => RUN_STATE_LABEL[activeSim.value.runState]);

const algorithmLabel = computed(() => {
  const kind = activeSim.value.algorithm;
  if (kind === 'qlearning') return 'Q-Learning';
  if (kind === 'sarsa') return 'Sarsa';
  return '未运行';
});
</script>

<template>
  <div class="page">
    <header class="topbar">
      <div class="brand">
        <span class="brand__mark" aria-hidden="true">
          <span class="brand__ball" />
        </span>
        <div class="brand__text">
          <h1>FindBall / Snake</h1>
          <p>同一套 Q-Learning 与 Sarsa，两个小游戏</p>
        </div>
      </div>

      <div class="status">
        <span class="status__item">
          <i class="status__dot" :class="`status__dot--${activeSim.runState}`" />
          {{ statusLabel }}
        </span>
        <span class="status__item">{{ algorithmLabel }}</span>
        <span class="status__item">
          回合 <b class="num">{{ activeSim.episode }}</b>
        </span>
      </div>
    </header>

    <nav
      class="tabs"
      role="tablist"
      :style="{ '--count': TABS.length, '--index': tabIndex }"
      aria-label="小游戏"
    >
      <span class="tabs__thumb" aria-hidden="true" />
      <button
        v-for="key in TABS"
        :key="key"
        type="button"
        role="tab"
        class="tabs__item"
        :class="{ 'tabs__item--active': tab === key }"
        :aria-selected="tab === key"
        @click="tab = key"
      >
        {{ TAB_LABEL[key] }}
      </button>
    </nav>

    <div class="tab-panel" :data-active="String(tab === 'findball')">
      <FindBallTab :sim="findBall" :env="findBallEnv" />
    </div>
    <div class="tab-panel" :data-active="String(tab === 'snake')">
      <SnakeTab :sim="snake" :env="snakeEnv" />
    </div>
  </div>
</template>

<style scoped>
.page {
  position: relative;
  z-index: 1;
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 20px 56px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.tab-panel[data-active='false'] {
  display: none;
}

/* 顶栏：品牌 + 状态条 */
.topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--border);
  animation: card-rise var(--dur-slow) var(--ease-out) both;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand__mark {
  position: relative;
  display: grid;
  place-items: center;
  width: 40px;
  height: 40px;
  border-radius: 12px;
  background:
    linear-gradient(rgba(255, 255, 255, 0.09) 1px, transparent 1px) 0 0 / 13px 13px,
    linear-gradient(90deg, rgba(255, 255, 255, 0.09) 1px, transparent 1px) 0 0 / 13px 13px,
    linear-gradient(160deg, rgba(34, 211, 238, 0.35), rgba(167, 139, 250, 0.28));
  border: 1px solid var(--border);
  box-shadow: 0 12px 30px -16px rgba(34, 211, 238, 0.9);
}

.brand__ball {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: radial-gradient(circle at 34% 28%, #9ec2ff, #2563eb 65%);
  box-shadow: 0 0 14px rgba(37, 99, 235, 0.95);
}

.brand__text h1 {
  font-size: 20px;
  font-weight: 650;
  letter-spacing: 0.01em;
}

.brand__text p {
  margin-top: 3px;
  font-size: 12px;
  color: var(--text-dim);
}

.status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.status__item {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 6px 12px;
  font-size: 12px;
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border);
  border-radius: 999px;
  transition:
    border-color var(--dur) var(--ease-out),
    color var(--dur) var(--ease-out);
}

.status__item:hover {
  color: var(--text);
  border-color: var(--border-strong);
}

.status__item b {
  color: var(--text);
  font-size: 13px;
}

.status__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--text-dim);
  transition: background var(--dur) linear;
}

.status__dot--running {
  background: var(--good);
  animation: topbar-pulse 1.4s ease-in-out infinite;
}

.status__dot--paused {
  background: var(--warn);
}

@keyframes topbar-pulse {
  0%,
  100% {
    box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(52, 211, 153, 0);
  }
}

/* 页签：滑块跟着选中项平移 */
.tabs {
  position: relative;
  display: flex;
  gap: 4px;
  width: fit-content;
  padding: 4px;
  background: rgba(0, 0, 0, 0.32);
  border: 1px solid var(--border);
  border-radius: 14px;
  animation: card-rise var(--dur-slow) var(--ease-out) both;
  animation-delay: 60ms;
}

.tabs__thumb {
  position: absolute;
  top: 4px;
  bottom: 4px;
  left: 4px;
  width: calc((100% - 8px) / var(--count));
  border-radius: 11px;
  background: linear-gradient(180deg, rgba(34, 211, 238, 0.92), rgba(34, 211, 238, 0.55));
  box-shadow: 0 10px 24px -14px rgba(34, 211, 238, 1);
  transform: translateX(calc(var(--index) * 100%));
  transition: transform var(--dur) var(--ease-spring);
}

.tabs__item {
  position: relative;
  z-index: 1;
  flex: 1 1 0;
  padding: 8px 26px;
  font: inherit;
  font-size: 13px;
  color: var(--text-muted);
  background: none;
  border: 0;
  border-radius: 11px;
  cursor: pointer;
  white-space: nowrap;
  transition: color var(--dur) var(--ease-out);
}

.tabs__item:hover {
  color: var(--text);
}

.tabs__item--active {
  color: #04121a;
  font-weight: 650;
}
</style>
