import type { Agent, StepResult } from './agent';
import type { FindBallEnv, State } from './env';

export type TickKind = 'step' | 'episode-end';

export interface TickResult {
  /** step = 本 tick 走了一步；episode-end = 球在终点/陷阱上，本 tick 把它送回起点 */
  type: TickKind;
  /** 本 tick 结束后球所在的状态 */
  state: State;
  /** 已完成的回合数 */
  episode: number;
  /** 累计步数 */
  steps: number;
  /** type === 'step' 时存在，便于界面/测试查看这一步到底做了什么 */
  transition?: StepResult;
  /** type === 'episode-end' 时存在：球停在哪个终止格上（界面拿它播命中动画） */
  endedAt?: State;
  /** type === 'episode-end' 时存在：是到终点还是踩了陷阱 */
  endReason?: 'goal' | 'trap';
}

/** 一集结束时的记录，供学习曲线与统计使用（纯粹是观测数据，不参与算法） */
export interface EpisodeRecord {
  /** 第几集，从 1 开始 */
  index: number;
  /** 这一集走了多少步 */
  steps: number;
  /** 这一集累计拿到多少收益 */
  reward: number;
  /** 收尾方式：到终点 / 踩陷阱 */
  end: 'goal' | 'trap';
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
 * 这里逐条对应，只是把两种算法统一到 Agent 接口上。
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

  private episodeSteps = 0;
  private episodeReward = 0;

  constructor(
    readonly env: FindBallEnv,
    readonly agent: Agent,
  ) {
    this.state = env.start;
  }

  /** 对应 qlStart / sarsaStart：球回到起点；Sarsa 在这里挑出第一个动作 */
  start(): void {
    this.state = this.env.start;
    this.agent.reset(this.state, this.env);
  }

  /**
   * 对应右键菜单的 "initialize blue ball's position"：只把球挪回起点。
   *
   * 与原实现的一处刻意差异：这里顺手让 agent 重新初始化动作。原实现只挪球，
   * Sarsa 会继续用"在上一个状态挑好的动作"（例如在 (0,0) 用 UP），一步就走出棋盘。
   * 结果上更容易理解，且不会踩到那个越界隐患。
   */
  resetBall(): void {
    this.state = this.env.start;
    this.agent.reset(this.state, this.env);
  }

  tick(): TickResult {
    if (this.env.isTerminal(this.state)) {
      const endedAt = this.state;
      const endReason = endedAt === this.env.goal ? 'goal' : 'trap';
      this.recordEpisode(endReason);
      this.state = this.env.start;
      this.episode += 1;
      this.agent.reset(this.state, this.env);
      return {
        type: 'episode-end',
        state: this.state,
        episode: this.episode,
        steps: this.steps,
        endedAt,
        endReason,
      };
    }

    const transition = this.agent.step(this.state, this.env);
    this.state = transition.next;
    this.steps += 1;
    this.episodeSteps += 1;
    this.episodeReward += transition.reward;
    this.lastTransition = transition;

    return { type: 'step', state: this.state, episode: this.episode, steps: this.steps, transition };
  }

  private recordEpisode(end: 'goal' | 'trap'): void {
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
