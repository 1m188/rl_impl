# rl_impl

简单的强化学习算法实现

## 两个游戏

两个页签共用同一份 `QLearning` / `Sarsa` / `Trainer`，差别只在"环境怎么定义状态"：

- **找红球**：状态就是格子编号（16 个），与 Python 原实现逐位对拍。
- **贪吃蛇**：状态压缩成蛇头局部局势的特征键 —— 周围 8 格危险位 + 食物相对方位 + 朝向，
  约 9216 个状态；奖励用"距离变化量"塑形（靠近 +1 / 远离 −1 / 吃食物 +10 / 撞死 −10），
  每集 300 步上限；默认 α=0.5、γ=0.9、ε=0.1。

## 跑起来

```bash
cd web
npm install
npm run dev        # 开发服务器
npm run build      # 类型检查 + 打包
npm test           # 测试
npm run golden     # 生成对拍数据
```

## 目录结构

```
web/
├── src/
│   ├── domain/                  # 纯 TypeScript，不 import 任何 vue，可在 node 里直接测
│   │   ├── actions.ts           # Action 类型与固定动作顺序
│   │   ├── game.ts              # GameEnv 接口：算法只依赖它，两个游戏都实现它
│   │   ├── env.ts               # 找红球的规则（FindBallEnv）
│   │   ├── snake.ts             # 贪吃蛇的规则与压缩状态编码（SnakeEnv）
│   │   ├── agent.ts             # Agent 接口、Q 表类型、argmax / maxValue / 随机取键
│   │   ├── qLearning.ts         # Q-Learning
│   │   ├── sarsa.ts             # Sarsa
│   │   ├── rng.ts               # 可注入的随机数来源（Math.random / 种子 LCG）
│   │   ├── trainer.ts           # 回合驱动：终止判定、复位、单步推进、每集记录、步数上限
│   │   └── qTableView.ts        # 找红球 Q 表的只读视图（策略、热力归一化）
│   ├── ui/heat.ts               # 热力配色（棋盘与 Q 表数值表共用）
│   ├── ui/labels.ts             # 收尾原因的中文标签
│   ├── composables/useSimulation.ts   # 把 domain 接到 Vue：固定步长时钟 + 响应式状态
│   ├── components/FindBallTab.vue     # 找红球页签
│   ├── components/SnakeTab.vue        # 贪吃蛇页签
│   ├── components/FindBallBoard.vue   # 4x4 网格、陷阱、终点、蓝球 + 策略箭头与热力叠加
│   ├── components/SnakeBoard.vue      # 8x8 网格、蛇身、食物、命中反馈
│   ├── components/SnakeStatePanel.vue # AI 视角：蛇头 3x3 视野 + 四个动作价值
│   ├── components/ControlPanel.vue    # 运行控制 / 倍速 / 快进 / 参数与种子 / 状态
│   ├── components/LearningCurve.vue   # 学习曲线（零依赖 SVG，带十字准线读数）
│   ├── components/StepDetail.vue      # 最近一步的 s/a/r/s' 与 Q 值更新
│   ├── components/FindBallQTablePanel.vue  # 找红球 Q 表数值表
│   ├── components/SnakeQTablePanel.vue     # 贪吃蛇 Q 表列表
│   └── App.vue / main.ts / style.css
├── tests/                       # vitest
│   ├── golden/parity.json       # 黄金轨迹
│   ├── parity.spec.ts           # 找红球逐位对拍
│   ├── qlearning.spec.ts        # 公式与 ε-greedy 的单测
│   ├── sarsa.spec.ts            # 含 on-policy / off-policy 的差异断言
│   ├── convergence.spec.ts      # 8 个种子 x 两种算法都要学会走安全最短路
│   ├── trainer.spec.ts          # 每集记录 / 单步细节 / 复位
│   ├── qtableView.spec.ts       # 策略、终止格、热力归一化
│   ├── snake.spec.ts            # 贪吃蛇机制：移动 / 吃 / 撞墙 / 咬到自己 / 状态编码 / 截断
│   ├── snakeLearning.spec.ts    # 贪吃蛇学得会：得分与存活步数都要涨
│   ├── ui.spec.ts               # 找红球界面
│   ├── ui-viz.spec.ts           # 暂停 / 单步 / 倍速 / 快进 / 种子可复现 / 叠加层
│   └── ui-snake.spec.ts         # 页签切换、两个游戏互不干扰、蛇的界面与 AI 视角
└── tools/gen_golden.py          # 生成 tests/golden/parity.json
```

分层原则：`domain/` 不依赖 Vue。这样算法能在 node 里跑测试，可视化面板也不会碰到算法。
