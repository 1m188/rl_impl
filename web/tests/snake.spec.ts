import { describe, expect, it } from 'vitest';

import { QLearning } from '../src/domain/qLearning';
import { lcgSource, scriptedSource } from '../src/domain/rng';
import { createSnakeEnv, SNAKE_REWARDS, TERMINAL_STATE } from '../src/domain/snake';
import { Trainer } from '../src/domain/trainer';

/** 食物永远落在第一个空格 —— 用来构造确定性的局面 */
const foodAtFirstFree = { next: () => 0 };

describe('贪吃蛇环境', () => {
  it('开局：长度 3、头在中间、朝右，食物不在蛇身上', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    env.reset();

    const snap = env.snapshot();
    expect(snap.body).toHaveLength(3);
    expect(env.x(snap.head)).toBe(4);
    expect(env.y(snap.head)).toBe(4);
    expect(snap.direction).toBe('RIGHT');
    expect(snap.dead).toBe(false);
    expect(snap.score).toBe(0);
    expect(snap.body).not.toContain(snap.food);
  });

  it('合法动作里不含反向（不能直接掉头）', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    const state = env.reset();
    expect(env.actionSet(state)).toStrictEqual(['UP', 'DOWN', 'RIGHT']);

    env.next(state, 'UP');
    // 朝上之后不能往下走
    expect(env.actionSet(TERMINAL_STATE === -1 ? env.snapshot().head : state)).not.toContain('DOWN');
  });

  it('走一步：头往前一格，身体跟着走，长度不变（没吃到食物）', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    const start = env.reset();
    const before = env.snapshot();

    const next = env.next(start, 'RIGHT');
    const after = env.snapshot();

    expect(env.x(after.head)).toBe(env.x(before.head) + 1);
    expect(after.body).toHaveLength(3);
    expect(after.body[0]).toBe(after.head);
    expect(after.body[1]).toBe(before.head); // 原来的头变成第二节

    // 压缩状态的关键性质：局面挪了一格，但"蛇头局部局势"没变（西边仍是身体、
    // 其余方向仍空着、食物仍在左上方、朝向仍是右），所以状态键是同一个。
    // 正因为大量不同局面共用同一个状态，Q 表才小到能被表格法学会。
    expect(next).toBe(start);
  });

  it('吃到食物：长度 +1、得分 +1、奖励 +10，并且立刻刷新食物', () => {
    const env = createSnakeEnv();
    // 0.56 -> 食物落在 (5,4)，正好在蛇头正前方；0.57 -> 吃完之后刷新到 (6,4)
    env.useRandom(scriptedSource([0.56, 0.57]));
    const start = env.reset();
    expect(env.snapshot().food).toBe(env.pack(5, 4));

    const next = env.next(start, 'RIGHT');
    const after = env.snapshot();

    expect(after.score).toBe(1);
    expect(after.body).toHaveLength(4);
    expect(env.reward(start, 'RIGHT')).toBe(SNAKE_REWARDS.food);
    expect(after.food).toBe(env.pack(6, 4));
    expect(next).not.toBe(TERMINAL_STATE);
  });

  it('塑形奖励 = 距离变化量：靠近 +1、远离 −1', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree); // 食物在 (0,0)
    const start = env.reset();

    // 头 (4,4) -> (5,4)：距离 8 -> 9，远离
    env.next(start, 'RIGHT');
    expect(env.reward(start, 'RIGHT')).toBe(-1);

    // 头 (5,4) -> (5,3)：距离 9 -> 8，靠近
    const second = env.next(TERMINAL_STATE, 'UP');
    expect(env.reward(TERMINAL_STATE, 'UP')).toBe(1);
    expect(second).not.toBe(TERMINAL_STATE);
  });

  it('撞墙：terminalReason = death、奖励 −10，状态变成终止态', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    let state = env.reset();

    // 一路向右直到撞墙（头从 x=4 到 x=7 需要 3 步，第 4 步撞墙）
    for (let i = 0; i < 3; i += 1) state = env.next(state, 'RIGHT');
    expect(env.x(env.snapshot().head)).toBe(7);
    expect(env.isTerminal(state)).toBe(false);

    const dead = env.next(state, 'RIGHT');
    expect(dead).toBe(TERMINAL_STATE);
    expect(env.terminalReason(dead)).toBe('death');
    expect(env.reward(state, 'RIGHT')).toBe(SNAKE_REWARDS.death);
    expect(env.snapshot().dead).toBe(true);
  });

  it('咬到自己：绕一圈撞上自己的身体', () => {
    const env = createSnakeEnv({ initialLength: 5 });
    env.useRandom(foodAtFirstFree);
    let state = env.reset(); // 身体 (4,4) (3,4) (2,4) (1,4) (0,4)，朝右

    state = env.next(state, 'RIGHT'); // 头 (5,4)
    state = env.next(state, 'DOWN'); // 头 (5,5)
    state = env.next(state, 'LEFT'); // 头 (4,5)
    expect(env.isTerminal(state)).toBe(false);

    state = env.next(state, 'UP'); // 头 (4,4) 撞回自己的身体
    expect(state).toBe(TERMINAL_STATE);
    expect(env.terminalReason(state)).toBe('death');
  });

  it('状态键的编码：初始局面的危险位与食物方位', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    const state = env.reset();

    const features = env.features(state);
    // 8 个方向里只有"西边"是自己的身体（蛇身朝左延伸）
    expect(features.dangerous).toStrictEqual([
      false, false, false,
      false, false, false,
      false, true,
    ]);
    // 食物 (0,0) 相对蛇头 (4,4)：在左上方
    expect(features.food).toStrictEqual({ dx: -1, dy: -1 });
    expect(features.direction).toBe('RIGHT');
    expect(features.terminal).toBe(false);

    // 危险位 bit7 + 方向 RIGHT(index 3) << 12
    expect(state).toBe(128 + (3 << 12));
    // 解码能还原出局面：同一个键代表同一类局部局势
    expect(env.features(state).key).toBe(state);
  });

  it('压缩后的状态数远小于整个棋盘的局面数', () => {
    // 14 位 -> 最多 16384 个键，实际可达约 256 种邻域 x 9 种食物方位 x 4 种朝向
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    const keys = new Set<number>();
    let state = env.reset();
    for (let i = 0; i < 200; i += 1) {
      const actions = env.actionSet(state);
      state = env.next(state, actions[i % actions.length]);
      if (state === TERMINAL_STATE) break;
      keys.add(state);
    }
    expect(keys.size).toBeLessThan(16384);
    expect(keys.size).toBeGreaterThan(0);
  });
});

describe('贪吃蛇 + 回合驱动器', () => {
  it('超过步数上限就截断，收尾原因是 timeout', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    const agent = new QLearning(0.5, 0.9, 0.1, lcgSource(7));
    const trainer = new Trainer(env, agent, { maxStepsPerEpisode: 3 });
    trainer.start();

    let reason: string | undefined;
    for (let i = 0; i < 10 && reason === undefined; i += 1) {
      const tick = trainer.tick();
      if (tick.type === 'episode-end') reason = tick.endReason;
    }
    expect(reason).toBe('timeout');
    expect(trainer.history[0].end).toBe('timeout');
    expect(trainer.history[0].steps).toBe(3);
  });

  it('一集走完会自动开新的一局（蛇回到初始长度）', () => {
    const env = createSnakeEnv();
    env.useRandom(foodAtFirstFree);
    const agent = new QLearning(0.5, 0.9, 0.1, lcgSource(7));
    const trainer = new Trainer(env, agent, { maxStepsPerEpisode: 3 });
    trainer.start();

    while (trainer.episode === 0) trainer.tick();
    expect(env.snapshot().body).toHaveLength(3);
    expect(env.snapshot().score).toBe(0);
    expect(trainer.history).toHaveLength(1);
  });
});
