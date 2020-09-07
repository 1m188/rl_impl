# rl_impl

针对具体的小游戏实现了一些强化学习算法，以作学习

使用python 3.x，pyside2 5.15.0

## Q-Learning

Q-Learning是一种model free，off policy，基于值的学习方法。其主要依赖于一个Q表，记录所有的状态和每个状态对应的策略以及收益。

游戏开始的时候首先初始化Q表，之后获取当前状态s，根据某种策略选择一个动作a（这里的策略可以是ε-greedy之类的），之后获取当前状态s采取该动作a的收益r和之后的新的状态s'，通过

`Q(s,a) = (1 - α) * Q(s,a) + α * [reward + γ * maxRQ(s')]`

公式进行Q表的更新。

maxRQ(s')指的是Q表之中s'状态所能够获得的最大收益。α指的是学习率，α越高则代表之前学习过的内容对于更新内容影响越小。γ代表折扣率，γ越高则代表对未来估量的经验值影响越大。之后更新状态s = s'。

反复进行上述的操作直到游戏结束，游戏结束后如果Q表没有收敛的话则再进行一遍游戏。

```
Initialize Q arbitrarily
Repeat (for each episode):
    Initialize s
    Repeat (for each step of episode):
        Choose a from s using policy derived from Q(ε-greedy)
        Take action a, observe r, s'
        Q(s,a) = (1 - α) * Q(s,a) + α * [reward + γ * maxRQ(s')]
        s = s'
    until s is terminal
```

Q-Learning最关键的部分就在于建立完整的Q表，然后通过查找Q表中的状态来获取可选择的动作以及每种动作所获取的收益以此做出抉择，就像是巴普洛夫的狗一样建立一种操作性条件反射。之所以说它是一种off policy的学习方法是因为每次更新Q表的时候明明有考虑到新状态能够获取的最大收益，但是下一次到新状态再选择动作时不一定选择的是能够带来最大收益的动作，因此它之前所做的都是一种模拟性质的行为，这就是off policy，不一定实际做出相应的动作。

## Sarsa

Sarsa和Q-Learning很相似，其核心都是采用Q表来做状态记录和查询，不同的是Q-Learning是off policy，而Sarsa是on policy。

其流程和Q-Learning也很相似。首先初始化Q表，然后游戏开始之后初始化当前状态s以及一个action a，这个action通过某种策略（ε-greedy）来选择，然后对于游戏中的每一步首先通过s和a来确定新的状态s'和收益r，之后通过某种策略选择新的状态s'下的动作a'，之后通过

`Q(s,a) = (1 - α) * Q(s,a) + α * [r + γ * Q(s',a')]`

公式更新Q表。

其中α和γ的意义和Q-Learning一样，这个公式和Q-Learning唯一不同的地方在于Q(s',a')，这个参数指新状态s'采取新动作a'所能够获取的收益，这和Q-Learning中新状态s'所能够获取的最大收益是不同的。最后更新状态和动作，s=s'，a=a'，这里和Q-Leanring也不同，这意味着每次用作更新状态s->s'时候所使用的动作不需要每次重新选择，直接使用传承下来的动作即可，而每次用来更新动作a的动作则是通过策略选择的使s'->s''的动作a'，而a'的所带来的收益为Q(s',a')，Q-Learning这里的收益是maxRQ(s')。

反复进行上述步骤直到游戏结束，如果Q表没有收敛则再进行一遍游戏。

```
Initialize Q arbitrarily
Repeat (for each episode):
    Initialize s
    Choose a from s using policy derived from Q(ε-greedy)
    Repeat (for each step of episode):
        Take action a, observe r, s'
        Choose a' from s' using policy derived from Q(ε-greedy)
        Q(s,a) = (1 - α) * Q(s,a) + α * [r + γ * Q(s',a')]
        s = s' ; a = a'
    until s is terminal
```

Sarsa在所选择的下一步动作的时候可能带有风险，而Q-Learning则总是选最大的；Sarsa选择了下一个状态的执行动作并且当状态转到下一个状态之后必定执行之前选择的动作，而Q-Learning则不一定，这也是为什么说Q-Learning是off policy而Sarsa是on policy的原因，Q-Learning用来判断当前这一步的依据它在下一步不一定执行，而Sarsa则很相信自己，它在这一步用来判断的依据下一步一定执行。