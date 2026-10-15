import { ACTION_ORDER, DELTA, type Action } from './actions';
import type { GameEnv, State } from './game';
import { mathRandomSource, type RandomSource } from './rng';

/**
 * 贪吃蛇环境。
 *
 * 状态怎么编码（这是这个游戏能不能用表格法学会的关键）
 * ---------------------------------------------------
 * 局面本身是"整条蛇身 + 食物位置"，直接拿它当状态的话状态数是天文级，
 * Q 表里每个状态基本只会被访问一次。所以这里把状态压缩成蛇头局部局势的特征键：
 *
 *   bit 0..7    蛇头周围 8 格的"危险位"（撞墙或咬到自己 = 1），顺序见 NEIGHBOR_OFFSETS
 *   bit 8..11   食物相对蛇头的方位（dx: 左/同/右，dy: 上/同/下，共 9 种）
 *   bit 12..13  当前朝向（0..3，对应 ACTION_ORDER）
 *
 * 一共 14 位、约 9216 个可达状态，Q 表小到可以直接列出来；配合下面这套密集奖励，
 * 几分钟就能看出它在追食物、会绕开自己的身体。
 *
 * 算法完全不知道这些：它只把 state 当成不透明的 Map 键，跟找红球用格子编号是一回事。
 *
 * 与环境的一个约定
 * ----------------
 * 这个环境是"有状态的模拟器"：蛇身、食物这些真实局面存在环境内部，
 * 传进 next() 的 state 只是当前局面的特征键。算法只会沿着真实轨迹调用
 * next(state, action)，所以这样是安全的；reward() 返回的也是上一步
 * next() 算好的收益（可以重复调用，结果不变）。
 */

/** 撞死或通关时返回的状态（正常特征键是 0..16383） */
export const TERMINAL_STATE: State = -1;

export const SNAKE_REWARDS = {
  /** 吃到食物 */
  food: 10,
  /** 撞墙 / 咬到自己 */
  death: -10,
} as const;

/**
 * 每步的塑形奖励 = 上一步到食物的曼哈顿距离 − 这一步的距离。
 * 也就是"靠近 +1、远离 −1、不变 0"。
 *
 * 这里特意没有照抄找红球的 −距离：那样等于每一步都在说"直接扑向食物"，
 * 蛇身挡路时它照样一头撞上去（实测平均只能活 12 步、吃 2 个食物）。
 * 改成距离变化量之后，"别撞死"这个信号不会被塑形压过去，
 * 同样的 2000 集里平均能活 49 步、吃 6 个食物。
 */

/** 蛇头周围 8 格，从左上开始顺时针 */
export const NEIGHBOR_OFFSETS: ReadonlyArray<readonly [number, number]> = [
  [-1, -1],
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
];

export interface SnakeConfig {
  width?: number;
  height?: number;
  /** 初始蛇长（含蛇头） */
  initialLength?: number;
}

export interface SnakeSnapshot {
  /** 蛇头所在格子 */
  head: State;
  /** 从蛇头到蛇尾的格子列表 */
  body: State[];
  /** 食物所在格子 */
  food: State;
  direction: Action;
  /** 吃到的食物个数 */
  score: number;
  dead: boolean;
  full: boolean;
}

export interface SnakeFeatures {
  key: State;
  /** 8 个方向的危险位，顺序同 NEIGHBOR_OFFSETS */
  dangerous: boolean[];
  /** 食物相对蛇头的方位（-1 / 0 / 1） */
  food: { dx: number; dy: number };
  direction: Action;
  terminal: boolean;
}

export interface SnakeEnv extends GameEnv {
  readonly width: number;
  readonly height: number;
  pack(x: number, y: number): State;
  x(state: State): number;
  y(state: State): number;
  /** 当前真实局面，供界面渲染 */
  snapshot(): SnakeSnapshot;
  /** 把一个特征键解码回人类可读的形式，供"AI 视角"面板与 Q 表列表使用 */
  features(state: State): SnakeFeatures;
  /** 贪吃蛇要随机撒食物，所以这里不是可选的（GameEnv 里是可选） */
  useRandom(rng: RandomSource): void;
}

const DEFAULT_CONFIG: Required<SnakeConfig> = {
  width: 8,
  height: 8,
  initialLength: 3,
};

function opposite(action: Action): Action {
  if (action === 'UP') return 'DOWN';
  if (action === 'DOWN') return 'UP';
  if (action === 'LEFT') return 'RIGHT';
  return 'LEFT';
}

export function createSnakeEnv(config: SnakeConfig = {}): SnakeEnv {
  const width = config.width ?? DEFAULT_CONFIG.width;
  const height = config.height ?? DEFAULT_CONFIG.height;
  const initialLength = config.initialLength ?? DEFAULT_CONFIG.initialLength;

  let rng: RandomSource = mathRandomSource;

  /** 从蛇头到蛇尾 */
  let body: State[] = [];
  let food = 0;
  let direction: Action = 'RIGHT';
  let dead = false;
  let full = false;
  let score = 0;
  let lastReward = 0;
  let terminalKind: 'death' | 'full' | null = null;

  const pack = (x: number, y: number): State => y * width + x;
  const x = (state: State): number => state % width;
  const y = (state: State): number => Math.floor(state / width);

  const inBounds = (cx: number, cy: number): boolean =>
    cx >= 0 && cx < width && cy >= 0 && cy < height;

  const manhattan = (a: State, b: State): number =>
    Math.abs(x(a) - x(b)) + Math.abs(y(a) - y(b));

  function spawnFood(): void {
    const free: State[] = [];
    for (let cell = 0; cell < width * height; cell += 1) {
      if (!body.includes(cell)) free.push(cell);
    }
    if (free.length === 0) {
      // 蛇把整个棋盘占满了 —— 通关
      food = -1;
      full = true;
      terminalKind = 'full';
      return;
    }
    food = free[Math.floor(rng.next() * free.length)];
  }

  /** 把当前局面压缩成状态键 */
  function featureKey(): State {
    const head = body[0];
    const hx = x(head);
    const hy = y(head);

    let key = 0;
    for (let i = 0; i < NEIGHBOR_OFFSETS.length; i += 1) {
      const [dx, dy] = NEIGHBOR_OFFSETS[i];
      const nx = hx + dx;
      const ny = hy + dy;
      const blocked = !inBounds(nx, ny) || body.includes(pack(nx, ny));
      if (blocked) key |= 1 << i;
    }

    const relX = x(food) < hx ? 0 : x(food) === hx ? 1 : 2;
    const relY = y(food) < hy ? 0 : y(food) === hy ? 1 : 2;
    key |= (relX * 3 + relY) << 8;
    key |= ACTION_ORDER.indexOf(direction) << 12;

    return key;
  }

  function reset(): State {
    body = [];
    const cy = Math.floor(height / 2);
    const headX = Math.floor(width / 2);
    for (let i = 0; i < initialLength; i += 1) {
      body.push(pack(headX - i, cy));
    }
    direction = 'RIGHT';
    dead = false;
    full = false;
    score = 0;
    lastReward = 0;
    terminalKind = null;
    spawnFood();
    return featureKey();
  }

  const actionSet = (_state: State): Action[] => {
    // 终止态也给一组动作：Q 表里它的值恒为 0，但不能出现"空行"（算法要取 max）
    if (dead || full) return [...ACTION_ORDER];
    // 不能直接掉头
    const forbidden = opposite(direction);
    return ACTION_ORDER.filter((action) => action !== forbidden);
  };

  const next = (_state: State, action: Action): State => {
    if (dead || full) return TERMINAL_STATE;

    const head = body[0];
    const nx = x(head) + DELTA[action][0];
    const ny = y(head) + DELTA[action][1];
    const previousDistance = manhattan(head, food);

    if (!inBounds(nx, ny)) {
      dead = true;
      terminalKind = 'death';
      lastReward = SNAKE_REWARDS.death;
      return TERMINAL_STATE;
    }

    const nextHead = pack(nx, ny);
    const eating = nextHead === food;
    // 不吃食物时尾巴会让位，所以尾格不算障碍
    const blocking = eating ? body : body.slice(0, -1);
    if (blocking.includes(nextHead)) {
      dead = true;
      terminalKind = 'death';
      lastReward = SNAKE_REWARDS.death;
      return TERMINAL_STATE;
    }

    body.unshift(nextHead);
    direction = action;

    if (eating) {
      score += 1;
      lastReward = SNAKE_REWARDS.food;
      spawnFood();
      // 吃掉最后一个空格 = 通关
      if (full) return TERMINAL_STATE;
    } else {
      body.pop();
      lastReward = previousDistance - manhattan(nextHead, food);
    }

    return featureKey();
  };

  const reward = (_state: State, _action: Action): number => lastReward;

  const isTerminal = (state: State): boolean => state === TERMINAL_STATE;

  const terminalReason = (state: State): string | null => {
    if (state !== TERMINAL_STATE) return null;
    return terminalKind ?? 'death';
  };

  const describe = (state: State): string => {
    if (state === TERMINAL_STATE) return terminalKind === 'full' ? '通关' : '撞死';
    const head = body[0];
    return `头(${x(head)},${y(head)}) 食物(${x(food)},${y(food)}) 长度${body.length}`;
  };

  const snapshot = (): SnakeSnapshot => ({
    head: body[0],
    body: [...body],
    food,
    direction,
    score,
    dead,
    full,
  });

  const features = (state: State): SnakeFeatures => {
    if (state === TERMINAL_STATE) {
      return {
        key: state,
        dangerous: NEIGHBOR_OFFSETS.map(() => false),
        food: { dx: 0, dy: 0 },
        direction,
        terminal: true,
      };
    }
    const dangerous = NEIGHBOR_OFFSETS.map((_, i) => ((state >> i) & 1) === 1);
    const foodCode = (state >> 8) & 0xf;
    const directionCode = (state >> 12) & 0x3;
    return {
      key: state,
      dangerous,
      food: { dx: (Math.floor(foodCode / 3) - 1), dy: (foodCode % 3) - 1 },
      direction: ACTION_ORDER[directionCode] ?? direction,
      terminal: false,
    };
  };

  return {
    width,
    height,
    pack,
    x,
    y,
    reset,
    actionSet,
    next,
    reward,
    isTerminal,
    terminalReason,
    describe,
    snapshot,
    features,
    useRandom: (source: RandomSource) => {
      rng = source;
    },
  };
}
