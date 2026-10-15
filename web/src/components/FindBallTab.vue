<script setup lang="ts">
import { computed, ref } from 'vue';

import type { SimulationApi } from '../composables/useSimulation';
import type { FindBallEnv } from '../domain/env';
import { readQTableView, type DisplayMode } from '../domain/qTableView';
import ControlPanel from './ControlPanel.vue';
import FindBallBoard from './FindBallBoard.vue';
import FindBallQTablePanel from './FindBallQTablePanel.vue';
import LearningCurve from './LearningCurve.vue';
import StepDetail from './StepDetail.vue';

const props = defineProps<{
  sim: SimulationApi;
  env: FindBallEnv;
}>();

const displayMode = ref<DisplayMode>('both');

/** Q 表本身不放进深层响应式，所以这里显式依赖 revision 来触发重算 */
const view = computed(() => {
  ref(props.sim.revision);
  return readQTableView(props.sim.agent, props.env);
});
</script>

<template>
  <div class="layout">
    <div class="col">
      <FindBallBoard
        :env="env"
        :state="sim.state"
        :view="view"
        :display-mode="displayMode"
        :trail="sim.trail"
        :burst="sim.lastEpisodeEnd"
        :move-duration-ms="sim.moveDurationMs"
        :teleported="sim.teleported"
        :running="sim.running"
      />
      <LearningCurve :history="sim.history" />
    </div>

    <div class="col">
      <ControlPanel
        :sim="sim"
        :display-mode="displayMode"
        reset-label="initialize blue ball's position"
        :state-total="env.width * env.height"
        @update:displayMode="displayMode = $event"
      />
      <StepDetail
        :transition="sim.lastTransition"
        :kind="sim.algorithm"
        :env="env"
        :alpha="sim.params.alpha"
        :gamma="sim.params.gamma"
      />
      <FindBallQTablePanel :view="view" :env="env" />
    </div>
  </div>
</template>
