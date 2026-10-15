"""用仓库里「未修改」的 algorithm/*.py 生成黄金轨迹，供 TS 侧逐位对拍。

为什么需要它
------------
把 Python 算法翻成 TypeScript 之后，"复刻得对不对"不能靠肉眼看。这里让原始
Python 代码在受控环境下跑一遍，把每一步的 (s, a, r, s') 和最终 Q 表导出成 JSON，
TS 侧用同一串随机数重放，逐位比较浮点数。

受控的两个地方（不修改算法文件，只改它看到的世界）
--------------------------------------------------
1. 随机数：原代码调用 random.random() 与 random.sample(ar.keys(), 1)。
   后者在 Python 3.11+ 已被禁用（dict_keys 不再当序列），所以这里把它换成
   同语义的 "在合法动作里等概率取一个"。两者都改为由 32 位 LCG 驱动，
   与 web/src/domain/rng.ts 的 lcgSource 逐位一致。
2. 动作顺序：原实现把动作放进 set，顺序随字符串 hash 每次运行都变。
   这里明确按 ACTION_ORDER 给出动作列表，让轨迹可复现。
   动作本身退化成字符串（算法只把它当字典的键用，不关心内容）。

动作没有暴露给任何回调，所以从状态位移反推动作（位移与动作一一对应）。
reward / updateState 则做了包装，记录的是「算法实际拿到的值」，而不是事后重算的。

用法：
    cd web && npm run golden        # 或 python tools/gen_golden.py
"""

from __future__ import annotations

import json
import math
import pathlib
import platform
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithm.q_learning import Q_Learning  # noqa: E402  （原文件，未做任何修改）
from algorithm.sarsa import Sarsa  # noqa: E402

# ---------------------------------------------------------------- 场景与超参
ACTION_ORDER = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

W = H = 4
START = (0, 0)
GOAL = (2, 2)
TRAPS = {(1, 1), (1, 2), (2, 1)}

ALPHA, GAMMA, EPSILON = 0.5, 0.5, 0.01
EPISODES = 30
# 上面 30 集里 ε=0.01 几乎不会触发探索分支，所以要再跑一段长训练：
# 它只记录步数/随机数消耗/最终 Q 表（不记录逐步轨迹，否则文件太大），
# 用来确认探索分支（随机动作）在两种实现里走的是同一条路。
LONG_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 2000  # 只是保险丝：踩到就说明轨迹失控，应当报错而不是悄悄截断

LCG = {"seed": 20240607, "a": 1664525, "c": 1013904223, "mod": 4294967296}
OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "tests" / "golden" / "parity.json"


# ------------------------------------------------------------ 受控的随机数
class ScriptedRandom:
    """与 web/src/domain/rng.ts 的 lcgSource 完全相同的 32 位 LCG。"""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF
        self.count = 0

    def next(self) -> float:
        self.count += 1
        self.state = (self.state * LCG["a"] + LCG["c"]) & 0xFFFFFFFF
        return self.state / LCG["mod"]

    def sample(self, population, k):  # 对应 random.sample(ar.keys(), 1)
        pop = list(population)  # dict_keys -> 插入顺序（= 动作顺序）
        if k != 1:
            raise AssertionError("原代码只会 k=1")
        return [pop[int(self.next() * len(pop))]]


# ------------------------------------------------------------------ 游戏规则
# 与 findball.py 的 getActionSet / getNewState / getReward 一致，
# 唯一的区别是动作集合有序（见文件头说明）。
def actionSet(pos):
    actions = []
    for name in ACTION_ORDER:
        dx, dy = DELTA[name]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < W and 0 <= ny < H:
            actions.append(name)
    return actions


def newState(pos, action):
    dx, dy = DELTA[action]
    return (pos[0] + dx, pos[1] + dy)


def reward(pos, action):
    nxt = newState(pos, action)
    if nxt == GOAL:
        return 10
    if nxt in TRAPS:
        return -10
    dx = nxt[0] - GOAL[0]
    dy = nxt[1] - GOAL[1]
    return -((dx**2 + dy**2) ** 0.5)


def infer_action(before, after):
    delta = (after[0] - before[0], after[1] - before[1])
    for name, value in DELTA.items():
        if value == delta:
            return name
    raise AssertionError(f"位移 {delta} 不对应任何动作")


# -------------------------------------------------------------------- 跑一集
def run(agent_cls, episodes: int):
    """按 findball.py 的 tick 模型跑：一个 tick 要么走一步，要么把球复位。

    这里刻意不用"每集一个 while 循环"的批量写法：那样最后一集结束后就退出了，
    而真实 GUI（以及 TS 的 Trainer）在下一 tick 还会执行一次"复位 + Sarsa initAction"。
    少算这一次会让 Sarsa 的随机数消耗正好差 1 个。
    """
    agent = agent_cls(ALPHA, GAMMA, EPSILON)
    holder = [START]
    step_log: list[dict] = []
    episode_log: list[dict] = []
    captured: dict = {}

    def logged_reward(pos, action):
        value = reward(pos, action)
        captured["reward"] = value  # 记录的是算法真正用到的那个数
        return value

    def logged_update_state(state):
        holder[0] = state

    if agent_cls is Sarsa:
        agent.initAction(holder[0], actionSet)  # 对应 sarsaStart

    completed = 0
    steps_in_episode = 0
    while completed < episodes:
        if holder[0] == GOAL or holder[0] in TRAPS:
            # 对应 qlStepRun / sarsaStepRun 的复位分支
            episode_log.append(
                {"steps": steps_in_episode, "end": "goal" if holder[0] == GOAL else "trap"}
            )
            steps_in_episode = 0
            holder[0] = START
            completed += 1
            if agent_cls is Sarsa:
                agent.initAction(holder[0], actionSet)
            continue

        before = holder[0]
        captured.clear()
        agent.stepRun(before, actionSet, newState, logged_reward, logged_update_state)
        after = holder[0]
        if "reward" not in captured:
            raise AssertionError("本步没有调用 getReward")
        step_log.append(
            {
                "s": list(before),
                "a": infer_action(before, after),
                "r": captured["reward"],
                "s2": list(after),
            }
        )
        steps_in_episode += 1
        if steps_in_episode >= MAX_STEPS_PER_EPISODE:
            raise AssertionError(f"第 {completed + 1} 集超过 {MAX_STEPS_PER_EPISODE} 步，轨迹失控")

    return agent, step_log, episode_log


def qtable_to_json(agent):
    """Q 表 -> {打包状态: {动作: 值}}，打包方式与 TS 侧一致：y * width + x"""
    table = {}
    for (x, y), row in agent.qtable.items():
        table[str(y * W + x)] = dict(row)
    return table


def main() -> None:
    # Math.sqrt 一定正确舍入；Python 的 ** 0.5 走 libm pow，这里先确认两者一致，
    # 否则 TS 侧用 Math.sqrt 就不可能逐位对拍。
    for value in (1, 2, 4, 5, 8, 9, 10, 13, 18):
        assert value**0.5 == math.sqrt(value), f"{value}**0.5 != math.sqrt({value})"

    # 让原算法用上受控的随机数。
    # 注意：每条算法轨迹都必须从同一个种子重新开始，否则第二条轨迹用的是
    # "被第一条消耗过的续流"，TS 侧就无法用同一个种子复现（这个坑踩过一次）。
    runs = {}
    for name, cls in (("qlearning", Q_Learning), ("sarsa", Sarsa)):
        scripted = ScriptedRandom(LCG["seed"])
        random.random = scripted.next  # type: ignore[assignment]
        random.sample = scripted.sample  # type: ignore[assignment]

        agent, step_log, episode_log = run(cls, EPISODES)
        runs[name] = {
            "steps": step_log,
            "episodes": episode_log,
            "finalQTable": qtable_to_json(agent),
            "draws": scripted.count,
        }

        # 长训练：同样从种子重新开始，覆盖探索分支
        scripted = ScriptedRandom(LCG["seed"])
        random.random = scripted.next  # type: ignore[assignment]
        random.sample = scripted.sample  # type: ignore[assignment]
        long_agent, long_steps, _ = run(cls, LONG_EPISODES)
        runs[name]["longRun"] = {
            "episodes": LONG_EPISODES,
            "steps": len(long_steps),
            "draws": scripted.count,
            "finalQTable": qtable_to_json(long_agent),
        }

        ends = {}
        for item in episode_log:
            ends[item["end"]] = ends.get(item["end"], 0) + 1
        print(
            f"{name}: {len(step_log)} 步 / {EPISODES} 集, 消耗随机数 {runs[name]['draws']} 个, "
            f"收尾统计 {ends}, Q 表 {len(agent.qtable)} 个状态"
        )
        print(
            f"{' ' * len(name)}  长训练 {LONG_EPISODES} 集: {len(long_steps)} 步, "
            f"消耗随机数 {runs[name]['longRun']['draws']} 个（含探索分支）"
        )

    golden = {
        "meta": {
            "generatedBy": "web/tools/gen_golden.py",
            "note": (
                "algorithm/q_learning.py 与 algorithm/sarsa.py 未做任何修改；"
                "只有 random 模块被替换为受控 LCG，动作集合改为有序列表"
            ),
            "python": platform.python_version(),
        },
        "actionOrder": ACTION_ORDER,
        "lcg": LCG,
        "config": {
            "width": W,
            "height": H,
            "start": list(START),
            "goal": list(GOAL),
            "traps": [list(t) for t in sorted(TRAPS)],
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon": EPSILON,
            "episodes": EPISODES,
        },
        "runs": runs,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(golden, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"已写入 {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
