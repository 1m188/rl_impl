<script setup lang="ts">
import { computed } from 'vue';

import type { SimulationApi } from '../composables/useSimulation';
import { ACTION_ORDER, type Action } from '../domain/actions';
import type { SnakeEnv } from '../domain/snake';
import ControlPanel from './ControlPanel.vue';
import LearningCurve from './LearningCurve.vue';
import SnakeBoard from './SnakeBoard.vue';
import SnakeQTablePanel from './SnakeQTablePanel.vue';
import SnakeStatePanel from './SnakeStatePanel.vue';
import StepDetail from './StepDetail.vue';

const props = defineProps<{
  sim: SimulationApi;
  env: SnakeEnv;
}>();

// Q 表与局面都不进深层响应式，统一用 revision 触发重算
const snapshot = computed(() => {
  void props.sim.revision;
  return props.env.snapshot();
});

const features = computed(() => {
  void props.sim.revision;
  return props.env.features(props.sim.state);
});

/** 当前状态在 Q 表里的那一行 */
const values = computed<Partial<Record<Action, number>> | null>(() => {
  void props.sim.revision;
  const row = props.sim.agent?.qtable.get(props.sim.state);
  if (!row) return null;
  const out: Partial<Record<Action, number>> = {};
  for (const action of ACTION_ORDER) out[action] = row.get(action);
  return out;
});

const agentTable = computed(() => {
  void props.sim.revision;
  return props.sim.agent?.qtable ?? null;
});
</script>

<template>
  <div class="layout">
    <div class="col">
      <SnakeBoard
        :env="env"
        :snapshot="snapshot"
        :state="sim.state"
        :features="features"
        :burst="sim.lastEpisodeEnd"
        :move-duration-ms="sim.moveDurationMs"
        :teleported="sim.teleported"
        :running="sim.running"
      />
      <SnakeStatePanel :features="features" :snapshot="snapshot" :values="values" />
      <LearningCurve :history="sim.history" />
    </div>

    <div class="col">
      <ControlPanel :sim="sim" />
      <StepDetail
        :transition="sim.lastTransition"
        :kind="sim.algorithm"
        :env="env"
        :alpha="sim.params.alpha"
        :gamma="sim.params.gamma"
      />
      <SnakeQTablePanel :env="env" :revision="sim.revision" :agent-table="agentTable" />
    </div>
  </div>
</template>
