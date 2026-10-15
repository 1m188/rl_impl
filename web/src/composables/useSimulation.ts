import { computed, onBeforeUnmount, reactive, ref, shallowRef } from 'vue';

import type { Agent, StepResult } from '../domain/agent';
import { createFindBallEnv, type FindBallEnv, type State } from '../domain/env';
import { QLearning } from '../domain/qLearning';
import { readQTableView, type QTableView } from '../domain/qTableView';
import { lcgSource, mathRandomSource, type RandomSource } from '../domain/rng';
import { Sarsa } from '../domain/sarsa';
import { Trainer, type EpisodeRecord } from '../domain/trainer';

export type AlgorithmKind = 'qlearning' | 'sarsa';

/** idle：没有算法；running：在跑；paused：保留 Q 表但不再推进（与 stop 的"丢弃"不同） */
export type RunState = 'idle' | 'running' | 'paused';

export interface SimulationParams {
  alpha: number;
  gamma: number;
  epsilon: number;
}

/** 超参照抄 findball.py 的 qlStart / sarsaStart：Q_Learning(0.5, 0.5, 0.01) */
export const DEFAULT_PARAMS: SimulationParams = { alpha: 0.5, gamma: 0.5, epsilon: 0.01 };

/** 原实现里 self.timer.start(50)：1x 就是每 50ms 走一步 */
const BASE_STEP_INTERVAL_MS = 50;
/** 标签页切到后台再回来时，不要一次性补上几百步；也限制了 20x 下单帧最多走几步 */
const MAX_CATCH_UP_STEPS = 40;
/** 学习曲线与统计只看最近这些集，避免把几万条记录都塞进响应式数据 */
const HISTORY_WINDOW = 600;

export const SPEEDS = [0.5, 1, 2, 5, 20] as const;
export type Speed = (typeof SPEEDS)[number];

/**
 * 把 domain 层接到 Vue 上。
 *
 * 逻辑推进用"固定步长累加器"：渲染跟着 requestAnimationFrame 走，
 * 但每一步的间隔由倍速决定（1x = 50ms，与原 QTimer 一致），与屏幕刷新率无关。
 */
export function useSimulation() {
  const env: FindBallEnv = createFindBallEnv();

  const ball = ref<State>(env.start);
  const runState = ref<RunState>('idle');
  const algorithm = ref<AlgorithmKind | null>(null);
  const episode = ref(0);
  const steps = ref(0);
  const speed = ref<Speed>(1);
  const params = reactive<SimulationParams>({ ...DEFAULT_PARAMS });
  const seedText = ref('');
  /** 本次运行实际用的种子：null 表示用 Math.random（不可复现） */
  const appliedSeed = ref<number | null>(null);
  const lastTransition = ref<StepResult | null>(null);
  const history = ref<EpisodeRecord[]>([]);
  /** 每 tick 递增，用来驱动 Q 表视图重算（Q 表本身不放进深层响应式） */
  const revision = ref(0);

  /**
   * Q 表放在浅引用里：20Hz 的热路径上没必要让 Vue 去深度代理它。
   * 视图通过 computed + revision 拿到摊平后的只读快照。
   */
  const agent = shallowRef<Agent | null>(null);
  let trainer: Trainer | null = null;
  let rafId = 0;
  let lastTimestamp = 0;
  let accumulator = 0;

  const stepIntervalMs = computed(() => BASE_STEP_INTERVAL_MS / speed.value);

  const tableView = computed<QTableView>(() => {
    revision.value; // 依赖：每个 tick 都变，保证 Q 表更新后重新摊平
    return readQTableView(agent.value, env);
  });

  const running = computed(() => runState.value === 'running');
  const hasAgent = computed(() => agent.value !== null);

  const seedStatus = computed<'random' | 'fixed' | 'invalid'>(() => {
    const raw = seedText.value.trim();
    if (raw === '') return 'random';
    return Number.isFinite(Number(raw)) ? 'fixed' : 'invalid';
  });

  function parseSeed(): number | null {
    if (seedStatus.value !== 'fixed') return null;
    return Math.trunc(Number(seedText.value.trim())) >>> 0;
  }

  function snapshot(): void {
    if (!trainer) return;
    ball.value = trainer.state;
    episode.value = trainer.episode;
    steps.value = trainer.steps;
    lastTransition.value = trainer.lastTransition;
    history.value = trainer.history.slice(-HISTORY_WINDOW);
    revision.value += 1;
  }

  function frame(timestamp: number): void {
    if (runState.value !== 'running' || !trainer) return;
    if (lastTimestamp === 0) lastTimestamp = timestamp;
    accumulator += timestamp - lastTimestamp;
    lastTimestamp = timestamp;

    let budget = MAX_CATCH_UP_STEPS;
    while (accumulator >= stepIntervalMs.value && budget > 0) {
      accumulator -= stepIntervalMs.value;
      budget -= 1;
      trainer.tick();
      snapshot();
    }
    if (budget === 0) accumulator = 0;

    rafId = requestAnimationFrame(frame);
  }

  function scheduleFrame(): void {
    accumulator = 0;
    lastTimestamp = 0;
    rafId = requestAnimationFrame(frame);
  }

  /** 对应右键菜单的 "Q-Learning start" / "Sarsa start" */
  function start(kind: AlgorithmKind): void {
    stop();

    const { alpha, gamma, epsilon } = params;
    const seed = parseSeed();
    const rng: RandomSource = seed === null ? mathRandomSource : lcgSource(seed);
    const next: Agent =
      kind === 'qlearning'
        ? new QLearning(alpha, gamma, epsilon, rng)
        : new Sarsa(alpha, gamma, epsilon, rng);

    agent.value = next;
    appliedSeed.value = seed;
    trainer = new Trainer(env, next);
    trainer.start(); // 对应 qlStart / sarsaStart：球回起点，Sarsa 在这里挑出第一个动作

    algorithm.value = kind;
    runState.value = 'running';
    snapshot();
    scheduleFrame();
  }

  /** 暂停：保留 Q 表与球的位置，之后可以继续 */
  function pause(): void {
    if (runState.value !== 'running') return;
    runState.value = 'paused';
    if (rafId !== 0) cancelAnimationFrame(rafId);
    rafId = 0;
  }

  function resume(): void {
    if (runState.value !== 'paused' || !trainer) return;
    runState.value = 'running';
    scheduleFrame();
  }

  /** 单步：只在暂停时可用，走一步（原实现没有这个能力，属于新增） */
  function stepOnce(): void {
    if (runState.value !== 'paused' || !trainer) return;
    trainer.tick();
    snapshot();
  }

  /** 快进训练：不等渲染，直接把 n 集跑完（高速看 Q 表收敛用） */
  function trainEpisodes(count: number): void {
    if (!trainer) return;
    const target = trainer.episode + count;
    let guard = count * 200 + 1000; // 每集正常只有几步，这只是防死循环的保险丝
    while (trainer.episode < target && guard > 0) {
      trainer.tick();
      guard -= 1;
    }
    snapshot();
  }

  /**
   * 对应右键菜单的 "stop"。
   * 与原实现一致：停下就丢掉这一轮学到的东西（原代码把 rlObj 置为 None），
   * 所以再次开始是从零学起，而不是接着上次继续。想保留请用 pause。
   */
  function stop(): void {
    if (rafId !== 0) cancelAnimationFrame(rafId);
    rafId = 0;
    runState.value = 'idle';
    agent.value = null;
    trainer = null;
    algorithm.value = null;
    appliedSeed.value = null;
    lastTransition.value = null;
    history.value = [];
    episode.value = 0;
    steps.value = 0;
    ball.value = env.start;
    revision.value += 1;
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

  onBeforeUnmount(() => {
    if (rafId !== 0) cancelAnimationFrame(rafId);
  });

  return reactive({
    env,
    ball,
    runState,
    running,
    hasAgent,
    algorithm,
    episode,
    steps,
    speed,
    params,
    seedText,
    seedStatus,
    appliedSeed,
    stepIntervalMs,
    lastTransition,
    history,
    tableView,
    start,
    pause,
    resume,
    stepOnce,
    trainEpisodes,
    stop,
    resetBall,
  });
}

/**
 * 界面拿到的仿真接口。
 * 用 reactive 包一层是为了让子组件里 `sim.ball` 这种嵌套访问也能自动解 ref
 * （Vue 只会在模板顶层解 ref，`sim.ball` 不解的话拿到的是 Ref 对象）。
 */
export type SimulationApi = ReturnType<typeof useSimulation>;
