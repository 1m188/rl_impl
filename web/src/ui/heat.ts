import { normalizeHeat } from '../domain/qTableView';

/**
 * 热力配色（展示层工具，Board 与 Q 表数值表共用）。
 *
 * 用"冷灰 → 琥珀 → 橙"的暖色渐变，并且刻意避开蓝色（球）与红色（终点）。
 * 只输出带透明度的颜色，这样叠在白底棋盘和暗底表格上都好看。
 */
export type Rgb = readonly [number, number, number];

const STOPS: ReadonlyArray<[number, Rgb]> = [
  [0, [148, 163, 184]],
  [0.55, [251, 191, 36]],
  [1, [249, 115, 22]],
];

export function heatRgb(ratio: number): Rgb {
  const clamped = Math.min(1, Math.max(0, ratio));
  for (let i = 1; i < STOPS.length; i += 1) {
    const [prevAt, prevColor] = STOPS[i - 1];
    const [at, color] = STOPS[i];
    if (clamped <= at || i === STOPS.length - 1) {
      const span = at - prevAt || 1;
      const t = Math.min(1, Math.max(0, (clamped - prevAt) / span));
      return [
        Math.round(prevColor[0] + (color[0] - prevColor[0]) * t),
        Math.round(prevColor[1] + (color[1] - prevColor[1]) * t),
        Math.round(prevColor[2] + (color[2] - prevColor[2]) * t),
      ];
    }
  }
  return STOPS[0][1];
}

/** value 在 [min, max] 里的相对位置 → rgba 颜色；alpha 从 alphaFrom 线性升到 alphaTo */
export function heatRgba(
  value: number,
  min: number,
  max: number,
  alphaFrom: number,
  alphaTo: number,
): string {
  const ratio = normalizeHeat(value, min, max);
  const [r, g, b] = heatRgb(ratio);
  const alpha = alphaFrom + ratio * (alphaTo - alphaFrom);
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}
