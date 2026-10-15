<script setup lang="ts">
import Board from './components/Board.vue';
import ControlPanel from './components/ControlPanel.vue';
import { HYPER_PARAMS, useSimulation } from './composables/useSimulation';

const { env, ball, running, algorithm, episode, steps, stateCount, start, stop, resetBall } =
  useSimulation();
</script>

<template>
  <main class="page">
    <header class="page__header">
      <h1>FindBall</h1>
      <p>
        蓝球从左上角出发去找红球，黑色方块是陷阱。两种算法都只靠一张 Q 表学习，
        按 ε-greedy 选动作，撞到终点或陷阱就重开一局。
      </p>
    </header>

    <section class="stage">
      <Board :env="env" :ball="ball" />
      <ControlPanel
        :running="running"
        :algorithm="algorithm"
        :episode="episode"
        :steps="steps"
        :state-count="stateCount"
        :alpha="HYPER_PARAMS.alpha"
        :gamma="HYPER_PARAMS.gamma"
        :epsilon="HYPER_PARAMS.epsilon"
        @start="start"
        @stop="stop"
        @reset-ball="resetBall"
      />
    </section>

    <footer class="page__footer">
      <p>
        这一版是把原来的 PySide2 程序
        （<code>findball.py</code> + <code>algorithm/q_learning.py</code> +
        <code>algorithm/sarsa.py</code>）搬到浏览器里：算法语义与超参一字未改，
        并且用原 Python 代码生成的黄金轨迹做了逐位对拍（见 <code>web/tests/</code>）。
      </p>
      <p>
        注意 stop 会丢掉这一轮学到的 Q 表（与原实现一致），再点 start 是从零学起。
      </p>
    </footer>
  </main>
</template>

<style scoped>
.page {
  max-width: 900px;
  margin: 0 auto;
  padding: 32px 20px 48px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.page__header h1 {
  margin: 0 0 8px;
  font-size: 24px;
  letter-spacing: 0.02em;
}

.page__header p,
.page__footer p {
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

.page__footer {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 13px;
}

code {
  padding: 1px 4px;
  background: #eee;
  border-radius: 4px;
  font-size: 12px;
}
</style>
