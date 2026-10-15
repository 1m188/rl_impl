<script setup lang="ts">
import { computed, ref } from 'vue';

import Board from './components/Board.vue';
import ControlPanel from './components/ControlPanel.vue';
import LearningCurve from './components/LearningCurve.vue';
import QTableViewPanel from './components/QTableViewPanel.vue';
import StepDetail from './components/StepDetail.vue';
import { useSimulation } from './composables/useSimulation';
import type { DisplayMode } from './domain/qTableView';

const sim = useSimulation();
const displayMode = ref<DisplayMode>('both');

const RUN_STATE_LABEL = {
  idle: '空闲',
  running: '运行中',
  paused: '已暂停',
} as const;

const statusLabel = computed(() => RUN_STATE_LABEL[sim.runState]);

const algorithmLabel = computed(() => {
  if (sim.algorithm === 'qlearning') return 'Q-Learning';
  if (sim.algorithm === 'sarsa') return 'Sarsa';
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
          <h1>FindBall</h1>
          <p>4×4 网格 · Q-Learning / Sarsa · 实时 Q 表可视化</p>
        </div>
      </div>

      <div class="status">
        <span class="status__item">
          <i class="status__dot" :class="`status__dot--${sim.runState}`" />
          {{ statusLabel }}
        </span>
        <span class="status__item">{{ algorithmLabel }}</span>
        <span class="status__item">
          回合 <b class="num">{{ sim.episode }}</b>
        </span>
      </div>
    </header>

    <div class="layout">
      <div class="col">
        <Board
          :env="sim.env"
          :ball="sim.ball"
          :view="sim.tableView"
          :display-mode="displayMode"
          :trail="sim.trail"
          :burst="sim.lastEpisodeEnd"
          :ball-duration-ms="sim.ballDurationMs"
          :teleported="sim.teleported"
          :running="sim.running"
        />
        <LearningCurve :history="sim.history" />
      </div>

      <div class="col">
        <ControlPanel
          :sim="sim"
          :display-mode="displayMode"
          @update:displayMode="displayMode = $event"
        />
        <StepDetail
          :transition="sim.lastTransition"
          :kind="sim.algorithm"
          :env="sim.env"
          :alpha="sim.params.alpha"
          :gamma="sim.params.gamma"
        />
        <QTableViewPanel :view="sim.tableView" :env="sim.env" />
      </div>
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

/* 仪表盘两栏：左边看过程，右边看数据与控制 */
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(360px, 0.92fr);
  gap: 18px;
  align-items: start;
}

.col {
  display: flex;
  flex-direction: column;
  gap: 18px;
  min-width: 0;
}

@media (max-width: 980px) {
  .layout {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
