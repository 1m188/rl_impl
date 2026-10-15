import { describe, expect, it } from 'vitest';

import { createFindBallEnv } from '../src/domain/env';
import { QLearning } from '../src/domain/qLearning';
import { lcgSource } from '../src/domain/rng';
import { Trainer } from '../src/domain/trainer';
import golden from './golden/parity.json';

const tuple = (pair: number[]): [number, number] => [pair[0], pair[1]];

function makeEnv() {
  const config = golden.config;
  return createFindBallEnv({
    width: config.width,
    height: config.height,
    start: tuple(config.start),
    goal: tuple(config.goal),
    traps: config.traps.map(tuple),
  });
}

describe('Trainer 的历史记录与单步细节', () => {
  it('每集记录与黄金数据完全一致：步数、收尾方式、以及该集累计收益', () => {
    const config = golden.config;
    const env = makeEnv();
    const agent = new QLearning(
      config.alpha,
      config.gamma,
      config.epsilon,
      lcgSource(golden.lcg.seed),
    );
    const trainer = new Trainer(env, agent);
    trainer.start();
    while (trainer.episode < config.episodes) trainer.tick();

    // 1) 步数与收尾方式
    expect(trainer.history.map(({ steps, end }) => ({ steps, end }))).toStrictEqual(
      golden.runs.qlearning.episodes,
    );

    // 2) 每集累计收益 == 黄金轨迹里那一集所有 reward 之和（累加顺序相同，可以逐位比较）
    let cursor = 0;
    for (const [index, episode] of golden.runs.qlearning.episodes.entries()) {
      const slice = golden.runs.qlearning.steps.slice(cursor, cursor + episode.steps);
      cursor += episode.steps;
      const sum = slice.reduce((total, step) => total + step.r, 0);
      expect(trainer.history[index].reward).toBe(sum);
    }
    expect(cursor).toBe(golden.runs.qlearning.steps.length);
  });

  it('lastTransition 记录的就是最后走过的那一步', () => {
    const config = golden.config;
    const env = makeEnv();
    const agent = new QLearning(
      config.alpha,
      config.gamma,
      config.epsilon,
      lcgSource(golden.lcg.seed),
    );
    const trainer = new Trainer(env, agent);
    trainer.start();
    while (trainer.episode < 5) trainer.tick();

    const last = trainer.lastTransition;
    expect(last).not.toBeNull();
    if (!last) return;
    expect(env.x(last.state)).toBeGreaterThanOrEqual(0);
    expect(last.next).toBe(env.next(last.state, last.action));
    expect(last.reward).toBe(env.reward(last.state, last.action));
    expect(last.qAfter).toBe(
      (1 - config.alpha) * last.qBefore + config.alpha * last.tdTarget,
    );
  });

  it('resetBall 把球送回起点，且不影响已完成的回合数', () => {
    const env = makeEnv();
    const agent = new QLearning(0.5, 0.5, 0.01, lcgSource(7));
    const trainer = new Trainer(env, agent);
    trainer.start();

    for (let i = 0; i < 3; i += 1) trainer.tick();
    const episodesBefore = trainer.episode;
    expect(trainer.state).not.toBe(env.start);

    trainer.resetBall();
    expect(trainer.state).toBe(env.start);
    expect(trainer.episode).toBe(episodesBefore);
  });

  it('球停在终点/陷阱上时，下一次 tick 只会复位并记一集，不会再多走一步', () => {
    const env = makeEnv();
    const agent = new QLearning(0.5, 0.5, 0.01, lcgSource(3));
    const trainer = new Trainer(env, agent);
    trainer.start();

    let guard = 0;
    while (!env.isTerminal(trainer.state)) {
      trainer.tick();
      guard += 1;
      expect(guard).toBeLessThan(200);
    }

    const stepsBeforeReset = trainer.steps;
    const tick = trainer.tick();
    expect(tick.type).toBe('episode-end');
    expect(trainer.steps).toBe(stepsBeforeReset); // 复位那一 tick 不算步数
    expect(trainer.episode).toBe(1);
    expect(trainer.state).toBe(env.start);
    expect(trainer.history).toHaveLength(1);
  });
});
