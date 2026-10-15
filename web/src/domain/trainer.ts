import type { Agent, StepResult } from './agent';
import type { GameEnv, State } from './game';

export type TickKind = 'step' | 'episode-end';

export interface TickResult {
  /** step = 本 tick 走了一步；episode-end = 本 tick 结束了一集并开了新的一局 */
  type: TickKind;
  /** 本 tick 结束后当前状态 */
  state: State;
  /** 已完成的回合数 */
  episode: number;
  /** 累计步数 */
  steps: number;
  /** type === 'step' 时存在，便于界面/测试查看这一步到底做了什么 */
  transition?: StepResult;
  /** type === 'episode-end' 时存在：结束在哪（界面拿它播命中动画） */
  endedAt?: State;
  /** type === 'episode-end' 时存在：收尾原因（goal / trap / death / full / timeout） */
  endReason?: string;
}

/** 一集结束时的记录，供学习曲线与统计使用（纯粹是观测数据，不参与算法） */
export interface EpisodeRecord {
  /** 第几集，从 1 开始 */
  index: number;
  /** 这一集走了多少步 */
  steps: number;
  /** 这一集累计拿到多少收益 */
  reward: number;
  /** 收尾原因 */
  end: string;
}

export interface TrainerOptions {
  /**
   * 一集最多走多少步，超过就截断（记为 timeout）。
   * 找红球不需要（原实现没有上限）；贪吃蛇必须要有，否则坏策略能绕一辈子。
   */
  maxStepsPerEpisode?: number;
}

/**
 * 历史记录上限：高速训练（20x）下每集约 6 步，跑十几分钟就是十万集，
 * 不设上限会一直吃内存。超出后按块丢弃最早的记录（摊还下来不影响性能）。
 */
const MAX_HISTORY = 50_000;
const TRIM_SIZE = 1_000;

/**
 * 回合驱动器 —— 对应 findball.py 里 qlStepRun / sarsaStepRun 的职责：
 * 边界（球到了终点或陷阱就复位重来）、状态推进、以及"一集开始"的准备。
 *
 * 原来的 tick 逻辑：
 *   if 球在终点或陷阱: 球回起点; (Sarsa 还要 initAction); return
 *   else: rlObj.stepRun(...)
 * 这里逐条对应，只是把两种算法统一到 Agent 接口上，把游戏规则统一到 GameEnv 接口上，
 * 于是同一份驱动器能同时驱动找红球和贪吃蛇。
 *
 * 另外记录每集的步数/收益/收尾方式与最近一步的细节 —— 这些只用于可视化，
 * 不参与任何算法计算，所以不会影响与 Python 原实现的逐位对拍。
 */
export class Trainer {
  state: State;
  /** 已完成的回合数（原实现没有这个计数，是界面需要才加的） */
  episode = 0;
  /** 累计步数（同上，纯展示用，不影响算法） */
  steps = 0;
  /** 每集记录，供学习曲线使用 */
  readonly history: EpisodeRecord[] = [];
  /** 最近一步的 s/a/r/s'/Q 变化，供"单步细节"面板使用 */
  lastTransition: StepResult | null = null;

  private readonly maxSteps: number;
  private episodeSteps = 0;
  private episodeReward = 0;

  constructor(
    readonly env: GameEnv,
    readonly agent: Agent,
    options: TrainerOptions = {},
  ) {
    this.maxSteps = options.maxStepsPerEpisode ?? Number.POSITIVE_INFINITY;
    this.state = env.reset();
  }

  /**
   * 开始一轮：开新的一局；Sarsa 在这里挑出第一个动作。
   *
   * 注意这是使用契约的一部分：构造完 Trainer 之后必须先调用 start()（或 resetBall()）
   * 再 tick()。Sarsa 的 step() 要求"本步要执行的动作"已经被挑好，直接 tick() 会报错。
   */
  start(): void {
    this.resetBall();
  }

  /**
   * 重置当前这一局（找红球里对应右键菜单的 "initialize blue ball's position"）。
   *
   * 与原实现的一处刻意差异：这里顺手让 agent 重新初始化动作。原实现只挪球，
   * Sarsa 会继续用"在上一个状态挑好的动作"（例如在 (0,0) 用 UP），一步就走出棋盘。
   */
  resetBall(): void {
    this.state = this.env.reset();
    this.agent.reset(this.state, this.env);
  }

  tick(): TickResult {
    const reason = this.env.terminalReason(this.state);
    if (reason !== null) return this.endEpisode(reason);

    // 截断：坏策略下贪吃蛇可能一直绕圈，需要一个步数上限
    if (this.episodeSteps >= this.maxSteps) return this.endEpisode('timeout');

    const transition = this.agent.step(this.state, this.env);
    this.state = transition.next;
    this.steps += 1;
    this.episodeSteps += 1;
    this.episodeReward += transition.reward;
    this.lastTransition = transition;

    return { type: 'step', state: this.state, episode: this.episode, steps: this.steps, transition };
  }

  private endEpisode(reason: string): TickResult {
    const endedAt = this.state;
    this.recordEpisode(reason);
    this.state = this.env.reset();
    this.episode += 1;
    this.agent.reset(this.state, this.env);

    return {
      type: 'episode-end',
      state: this.state,
      episode: this.episode,
      steps: this.steps,
      endedAt,
      endReason: reason,
    };
  }

  private recordEpisode(end: string): void {
    this.history.push({
      index: this.episode + 1,
      steps: this.episodeSteps,
      reward: this.episodeReward,
      end,
    });
    this.episodeSteps = 0;
    this.episodeReward = 0;

    if (this.history.length > MAX_HISTORY) {
      this.history.splice(0, TRIM_SIZE);
    }
  }
}
