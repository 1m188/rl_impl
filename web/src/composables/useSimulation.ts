import { computed, onBeforeUnmount, reactive, ref, shallowRef, watch, type Ref } from 'vue';

import type { Agent, StepResult } from '../domain/agent';
import type { GameEnv, State } from '../domain/game';
import { QLearning } from '../domain/qLearning';
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

/** 找红球的超参：照抄 findball.py 的 Q_Learning(0.5, 0.5, 0.01) */
export const DEFAULT_PARAMS: SimulationParams = { alpha: 0.5, gamma: 0.5, epsilon: 0.01 };

/** 原实现里 self.timer.start(50)：1x 就是每 50ms 走一步 */
const BASE_STEP_INTERVAL_MS = 50;
/** 标签页切到后台再回来时，不要一次性补上几百步；也限制了 20x 下单帧最多走几步 */
const MAX_CATCH_UP_STEPS = 40;
/** 学习曲线与统计只看最近这些集，避免把几万条记录都塞进响应式数据 */
const HISTORY_WINDOW = 600;
/** 轨迹残影保留几个点 */
const TRAIL_LENGTH = 6;

export const SPEEDS = [0.5, 1, 2, 5, 20] as const;
export type Speed = (typeof SPEEDS)[number];

export interface SimulationOptions {
  /** 一集最多走多少步（贪吃蛇需要，找红球不需要） */
  maxStepsPerEpisode?: number;
  /** 默认超参，不同游戏不一样 */
  defaultParams?: SimulationParams;
  /** 所在页签是否激活：不激活时停掉动画帧但保留状态 */
  active?: Ref<boolean>;
}

/**
 * 把 domain 层接到 Vue 上。与具体游戏无关：只依赖 GameEnv 接口，
 * 所以找红球和贪吃蛇共用同一份实现。
 *
 * 逻辑推进用"固定步长累加器"：渲染跟着 requestAnimationFrame 走，
 * 但每一步的间隔由倍速决定（1x = 50ms，与原 QTimer 一致），与屏幕刷新率无关。
 */
export function useSimulation(env: GameEnv, options: SimulationOptions = {}) {
  const isActive = options.active ?? ref(true);

  const state = ref<State>(env.reset());
  const runState = ref<RunState>('idle');
  const algorithm = ref<AlgorithmKind | null>(null);
  const episode = ref(0);
  const steps = ref(0);
  const knownStates = ref(0);
  const speed = ref<Speed>(1);
  const params = reactive<SimulationParams>({ ...(options.defaultParams ?? DEFAULT_PARAMS) });
  const seedText = ref('');
  /** 本次运行实际用的种子：null 表示用 Math.random（不可复现） */
  const appliedSeed = ref<number | null>(null);
  const lastTransition = ref<StepResult | null>(null);
  const history = ref<EpisodeRecord[]>([]);
  /** 刚走过的几个状态（最新的在前），用来画轨迹残影 */
  const trail = ref<State[]>([]);
  /** 刚发生"换局瞬移"：这一步是直接闪回开局位置，不该滑过去 */
  const teleported = ref(false);
  /** 最近一次一集结束的位置与原因，key 是回合数（用于让命中动画重新播放） */
  const lastEpisodeEnd = ref<{ key: number; state: State; reason: string } | null>(null);
  /** 每 tick 递增，用来驱动 Q 表视图重算（Q 表本身不放进深层响应式） */
  const revision = ref(0);

  /**
   * Q 表放在浅引用里：20Hz 的热路径上没必要让 Vue 去深度代理它。
   * 各游戏自己用 computed + revision 把它摊平成需要的视图。
   */
  const agent = shallowRef<Agent | null>(null);
  let trainer: Trainer | null = null;
  let rafId = 0;
  let lastTimestamp = 0;
  let accumulator = 0;

  const stepIntervalMs = computed(() => BASE_STEP_INTERVAL_MS / speed.value);

  /**
   * 移动动画时长：略短于一步的间隔，这样每步之间能滑过去又不至于"追不上"下一步。
   * 20x 时只有 2ms，实际上就是瞬移 —— 本来也看不清。
   */
  const moveDurationMs = computed(() =>
    Math.round(Math.min(Math.max(stepIntervalMs.value * 0.85, 0), 150)),
  );

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
    state.value = trainer.state;
    episode.value = trainer.episode;
    steps.value = trainer.steps;
    knownStates.value = trainer.agent.qtable.size;
    lastTransition.value = trainer.lastTransition;
    history.value = trainer.history.slice(-HISTORY_WINDOW);
    revision.value += 1;
  }

  /**
   * 推进一个 tick，并把"给动画看的东西"一起更新：
   * 走一步就延长轨迹残影，一集结束就记下结束位置、清空残影、并标记这一次是瞬移。
   *
   * 残影按"最新在前"存，界面上就能用固定的下标当 key，元素被复用而不是每步重建，
   * CSS 过渡才连得上。
   */
  function advance(): void {
    if (!trainer) return;
    teleported.value = false;
    const tick = trainer.tick();

    if (tick.type === 'episode-end') {
      trail.value = [];
      teleported.value = true;
      if (tick.endedAt !== undefined && tick.endReason) {
        lastEpisodeEnd.value = {
          key: tick.episode,
          state: tick.endedAt,
          reason: tick.endReason,
        };
      }
    } else if (trail.value[0] !== tick.state) {
      trail.value = [tick.state, ...trail.value].slice(0, TRAIL_LENGTH);
    }

    snapshot();
  }

  function frame(timestamp: number): void {
    if (runState.value !== 'running' || !trainer) return;
    // 页签被切走：停止调度动画帧，但保留状态（切回来会继续）
    if (!isActive.value) {
      rafId = 0;
      return;
    }
    if (lastTimestamp === 0) lastTimestamp = timestamp;
    accumulator += timestamp - lastTimestamp;
    lastTimestamp = timestamp;

    let budget = MAX_CATCH_UP_STEPS;
    while (accumulator >= stepIntervalMs.value && budget > 0) {
      accumulator -= stepIntervalMs.value;
      budget -= 1;
      advance();
    }
    if (budget === 0) accumulator = 0;

    rafId = requestAnimationFrame(frame);
  }

  function scheduleFrame(): void {
    if (!isActive.value) return;
    accumulator = 0;
    lastTimestamp = 0;
    rafId = requestAnimationFrame(frame);
  }

  watch(isActive, (active) => {
    if (active && runState.value === 'running' && rafId === 0) scheduleFrame();
  });

  /** 对应右键菜单的 "Q-Learning start" / "Sarsa start" */
  function start(kind: AlgorithmKind): void {
    stop();

    const { alpha, gamma, epsilon } = params;
    const seed = parseSeed();
    const rng: RandomSource = seed === null ? mathRandomSource : lcgSource(seed);

    // 环境自己的随机数（贪吃蛇撒食物）用另一条流，免得跟 agent 的探索数重叠
    if (env.useRandom) {
      env.useRandom(seed === null ? mathRandomSource : lcgSource((seed + 0x9e3779b9) >>> 0));
    }

    const next: Agent =
      kind === 'qlearning'
        ? new QLearning(alpha, gamma, epsilon, rng)
        : new Sarsa(alpha, gamma, epsilon, rng);

    agent.value = next;
    appliedSeed.value = seed;
    trainer = new Trainer(env, next, { maxStepsPerEpisode: options.maxStepsPerEpisode });
    trainer.start();

    algorithm.value = kind;
    runState.value = 'running';
    snapshot();
    scheduleFrame();
  }

  /** 暂停：保留 Q 表与局面，之后可以继续 */
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
    advance();
  }

  /** 快进训练：不等渲染，直接把 n 集跑完（高速看 Q 表收敛用） */
  function trainEpisodes(count: number): void {
    if (!trainer) return;
    const target = trainer.episode + count;
    let guard = count * 2000 + 1000; // 每集步数不定，这只是防死循环的保险丝
    while (trainer.episode < target && guard > 0) {
      advance();
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
    trail.value = [];
    teleported.value = false;
    lastEpisodeEnd.value = null;
    episode.value = 0;
    steps.value = 0;
    knownStates.value = 0;
    state.value = env.reset();
    revision.value += 1;
  }

  /** 重置当前这一局（找红球里对应右键菜单的 "initialize blue ball's position"） */
  function resetState(): void {
    if (!trainer) {
      state.value = env.reset();
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
    state,
    runState,
    running,
    hasAgent,
    algorithm,
    episode,
    steps,
    knownStates,
    speed,
    params,
    seedText,
    seedStatus,
    appliedSeed,
    stepIntervalMs,
    moveDurationMs,
    lastTransition,
    history,
    trail,
    teleported,
    lastEpisodeEnd,
    revision,
    agent,
    start,
    pause,
    resume,
    stepOnce,
    trainEpisodes,
    stop,
    resetState,
  });
}

/**
 * 界面拿到的仿真接口。
 * 用 reactive 包一层是为了让子组件里 `sim.state` 这种嵌套访问也能自动解 ref
 * （Vue 只会在模板顶层解 ref，`sim.state` 不解的话拿到的是 Ref 对象）。
 */
export type SimulationApi = ReturnType<typeof useSimulation>;
