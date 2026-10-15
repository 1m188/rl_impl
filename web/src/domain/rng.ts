/**
 * 随机数来源。
 *
 * 原实现直接用 random.random() / random.sample()。TS 侧把它变成可注入的接口，
 * 目的有两个：一是测试可以对同一串随机数复现同一段学习轨迹，二是阶段二想加
 * "种子" 输入框时不用动算法。
 */
export interface RandomSource {
  /** 返回 [0, 1) 之间的数，对应 Python 的 random.random() */
  next(): number;
}

export const mathRandomSource: RandomSource = {
  next: () => Math.random(),
};

/**
 * 32 位 LCG（Numerical Recipes 参数），用于复现性。
 * 与 tools/gen_golden.py 里的 Python 实现逐位一致：全程整数运算，
 * 用 Math.imul 保证 32 位乘法不丢精度（普通乘法会超过 2^53）。
 */
export const LCG_PARAMS = { a: 1664525, c: 1013904223, mod: 4294967296 } as const;

export function lcgSource(seed: number): RandomSource {
  let state = seed >>> 0;
  return {
    next: () => {
      state = (Math.imul(state, LCG_PARAMS.a) + LCG_PARAMS.c) >>> 0;
      return state / LCG_PARAMS.mod;
    },
  };
}

/** 预先给定的一串 [0,1) 数值，测试里用来精确控制"探索还是利用" */
export function scriptedSource(values: readonly number[]): RandomSource {
  let index = 0;
  return {
    next: () => {
      if (index >= values.length) throw new Error('scriptedSource: 预设的随机数已用尽');
      return values[index++];
    },
  };
}
