<script setup lang="ts">
import { ref } from 'vue';

import Board from './components/Board.vue';
import ControlPanel from './components/ControlPanel.vue';
import LearningCurve from './components/LearningCurve.vue';
import QTableViewPanel from './components/QTableViewPanel.vue';
import StepDetail from './components/StepDetail.vue';
import { useSimulation } from './composables/useSimulation';
import type { DisplayMode } from './domain/qTableView';

const sim = useSimulation();
const displayMode = ref<DisplayMode>('both');
</script>

<template>
  <main class="page">
    <header class="page__header">
      <h1>FindBall</h1>
      <p>
        蓝球从左上角出发去找红球，黑色方块是陷阱。两种算法都只靠一张 Q 表学习，按 ε-greedy 选动作，
        撞到终点或陷阱就重开一局。棋盘上的箭头与底色就是这个 Q 表的可视化。
      </p>
    </header>

    <div class="stage">
      <Board
        :env="sim.env"
        :ball="sim.ball"
        :view="sim.tableView"
        :display-mode="displayMode"
      />
      <ControlPanel
        :sim="sim"
        :display-mode="displayMode"
        @update:displayMode="displayMode = $event"
      />
    </div>

    <LearningCurve :history="sim.history" />

    <div class="details">
      <StepDetail
        :transition="sim.lastTransition"
        :kind="sim.algorithm"
        :env="sim.env"
        :alpha="sim.params.alpha"
        :gamma="sim.params.gamma"
      />
      <QTableViewPanel :view="sim.tableView" :env="sim.env" />
    </div>
  </main>
</template>

<style scoped>
.page {
  max-width: 1080px;
  margin: 0 auto;
  padding: 32px 20px 48px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page__header h1 {
  margin: 0 0 8px;
  font-size: 24px;
  letter-spacing: 0.02em;
}

.page__header p {
  margin: 0;
  color: #4a4a4a;
  line-height: 1.7;
  font-size: 14px;
}

.stage {
  display: flex;
  flex-wrap: wrap;
  gap: 28px;
  align-items: flex-start;
  padding: 20px;
  background: #fff;
  border: 1px solid #e2e2e2;
  border-radius: 10px;
}

.details {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-start;
}

.details > * {
  flex: 1 1 320px;
}
</style>
