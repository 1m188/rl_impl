import { onBeforeUnmount, ref, shallowRef } from 'vue';

import type { Agent } from '../domain/agent';
import { createFindBallEnv, type FindBallEnv, type State } from '../domain/env';
import { QLearning } from '../domain/qLearning';
import { mathRandomSource } from '../domain/rng';
import { Sarsa } from '../domain/sarsa';
import { Trainer } from '../domain/trainer';

export type AlgorithmKind = 'qlearning' | 'sarsa';

/**
 * 超参照抄 findball.py 的 qlStart / sarsaStart：Q_Learning(0.5, 0.5, 0.01)
 * 也就是 α=0.5（学习率）、γ=0.5（折扣率）、ε=0.01（ε-greedy 的探索率）。
 */
export const HYPER_PARAMS = { alpha: 0.5, gamma: 0.5, epsilon: 0.01 } as const;

/** 原实现里 self.timer.start(50)：每 50ms 走一步（另有 15ms 一次的窗口重绘） */
const STEP_INTERVAL_MS = 50;
/** 标签页切到后台再回来时，不要一次性补上几百步 */
const MAX_CATCH_UP_STEPS = 10;

/**
 * 把 domain 层接到 Vue 上。
 *
 * 逻辑推进用"固定步长累加器"：渲染跟着 requestAnimationFrame 走，
 * 但每一步的间隔恒为 50ms，与屏幕刷新率无关（原来的 QTimer 也是这个意思，
 * 只是这里顺带把倍速/单步留出了位置）。
 */
export function useSimulation() {
  const env: FindBallEnv = createFindBallEnv();

  const ball = ref<State>(env.start);
  const running = ref(false);
  const algorithm = ref<AlgorithmKind | null>(null);
  const episode = ref(0);
  const steps = ref(0);
  const stateCount = ref(0);

  /**
   * Q 表放在浅引用里：20Hz 的热路径上没必要让 Vue 去深度代理它。
   * 阶段二要把 Q 表画成热力图时，用 tick 结束后更新的 stateCount（或另加版本号）触发刷新即可。
   */
  const agent = shallowRef<Agent | null>(null);
  let trainer: Trainer | null = null;
  let rafId = 0;
  let lastTimestamp = 0;
  let accumulator = 0;

  function snapshot(): void {
    if (!trainer) return;
    ball.value = trainer.state;
    episode.value = trainer.episode;
    steps.value = trainer.steps;
    stateCount.value = trainer.agent.qtable.size;
  }

  function frame(timestamp: number): void {
    if (!running.value) return;
    if (lastTimestamp === 0) lastTimestamp = timestamp;
    accumulator += timestamp - lastTimestamp;
    lastTimestamp = timestamp;

    let budget = MAX_CATCH_UP_STEPS;
    while (accumulator >= STEP_INTERVAL_MS && budget > 0) {
      accumulator -= STEP_INTERVAL_MS;
      budget -= 1;
      trainer?.tick();
      snapshot();
    }
    if (budget === 0) accumulator = 0;

    rafId = requestAnimationFrame(frame);
  }

  /** 对应右键菜单的 "Q-Learning start" / "Sarsa start" */
  function start(kind: AlgorithmKind): void {
    stop();

    const { alpha, gamma, epsilon } = HYPER_PARAMS;
    const next: Agent =
      kind === 'qlearning'
        ? new QLearning(alpha, gamma, epsilon, mathRandomSource)
        : new Sarsa(alpha, gamma, epsilon, mathRandomSource);

    agent.value = next;
    trainer = new Trainer(env, next);
    trainer.start(); // 对应 qlStart / sarsaStart：球回起点，Sarsa 在这里挑出第一个动作

    algorithm.value = kind;
    running.value = true;
    accumulator = 0;
    lastTimestamp = 0;
    snapshot();

    rafId = requestAnimationFrame(frame);
  }

  /**
   * 对应右键菜单的 "stop"。
   * 与原实现一致：停下就丢掉这一轮学到的东西（原代码把 rlObj 置为 None），
   * 所以再次开始是从零学起，而不是接着上次继续。
   */
  function stop(): void {
    if (rafId !== 0) cancelAnimationFrame(rafId);
    rafId = 0;
    running.value = false;
    agent.value = null;
    trainer = null;
  }

  /** 对应右键菜单的 "initialize blue ball's position" */
  function resetBall(): void {
    if (!trainer) {
      ball.value = env.start;
      return;
    }
    trainer.resetBall();
    snapshot();
  }

  onBeforeUnmount(stop);

  return {
    env,
    ball,
    running,
    algorithm,
    episode,
    steps,
    stateCount,
    /** 阶段二画 Q 表热力图 / 策略箭头时从这里取数据 */
    agent,
    start,
    stop,
    resetBall,
  };
}
