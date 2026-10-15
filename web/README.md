# rl_impl

简单的强化学习算法实现

## 跑起来

```bash
cd web
npm install
npm run dev        # 开发服务器
npm run build      # 类型检查 + 打包
npm test           # 测试
```

## 目录结构

```
web/
├── src/
│   ├── domain/                  # 纯 TypeScript，不 import 任何 vue，可在 node 里直接测
│   │   ├── actions.ts           # Action 类型与固定动作顺序
│   │   ├── env.ts               # 游戏规则
│   │   ├── agent.ts             # Agent 接口、Q 表类型、argmax / maxValue / 随机取键
│   │   ├── qLearning.ts         # Q-Learning
│   │   ├── sarsa.ts             # Sarsa
│   │   ├── rng.ts               # 可注入的随机数来源（Math.random / 种子 LCG）
│   │   └── trainer.ts           # 回合驱动：终止判定、复位、单步推进
│   ├── composables/useSimulation.ts   # 把 domain 接到 Vue：固定步长时钟 + 响应式状态
│   ├── components/Board.vue           # 4x4 网格、陷阱、终点、蓝球
│   ├── components/ControlPanel.vue    # 原来的四个右键菜单项改成按钮
│   └── App.vue / main.ts / style.css
├── tests/                       # vitest
│   ├── golden/parity.json       # 黄金轨迹
│   ├── parity.spec.ts           # 逐位对拍
│   ├── qlearning.spec.ts        # 公式与 ε-greedy 的单测
│   ├── sarsa.spec.ts            # 含 on-policy / off-policy 的差异断言
│   ├── convergence.spec.ts      # 8 个种子 x 两种算法都要学会走安全最短路
│   └── ui.spec.ts               # 挂载 App，验证"点按钮 → 球动起来"
└── tools/gen_golden.py          # 生成 tests/golden/parity.json
```

分层原则：`domain/` 不依赖 Vue。这样算法能在 node 里跑测试，阶段二加可视化面板时也不用动它。
