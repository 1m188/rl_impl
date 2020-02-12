import random


# Q-Learning
# it's core is building operant conditioned reflex, and make agent take the best action according to the q table.
# 1. init state.
# 2. choose a action with ε-greedy poilcy, and get observer of s'(new state) and reward.
# 3. update q table: Q(s,a) = (1 - α) * Q(s,a) + α * [reward + γ * maxQ(s')].
# maxQ(s') means that the max reward can get in new state.
# α means learn rate, the α bigger, the experience learn before will keep less.
# γ means discount rate, the γ bigger, the future's influence will bigger.
# 4. take actual action, and update state.
# 5. repeat 2 3 4 until game end.
# 6. repeat steps above to training until q table convergence.
# Initialize Q arbitrarily
# Repeat (for each episode):
#     Initialize s
#     Repeat (for each step of episode):
#         Choose a from s using policy derived from Q(ε-greedy)
#         Take action a, observe r
#         Q(s,a) = (1 - α) * Q(s,a) + α * [reward + γ * maxQ(s')]
#         s = s'
#     until s is terminal
class Q_Learning:

    # alpha, gamma, epsilon are all >0 and <1
    def __init__(self, alpha: float, gamma: float, epsilon: float):

        # self.qtable's values are also dicts, {action:reward} means
        # the action can take in this state and the action's reward
        self.qtable = {}  # q table
        self.alpha = alpha  # learn rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # epsilon-greedy

    # initialize the current state, if the state has been never got
    # before, set the reward of all actions about the state are 0
    # actionSet is the action set that the state can take
    def initState(self, state, actionSet: set):
        if state not in self.qtable:
            p = {}
            for a in actionSet:
                p[a] = 0
            self.qtable[state] = p

    # get action with epsilon-greedy
    # state means current state
    def epsilon_greedy(self, state):
        rate = random.random()
        ar = self.qtable[state]
        return random.sample(ar.keys(), 1)[0] if rate < self.epsilon else max(ar, key=ar.get)

    # update q table
    # r means the reward that state take action
    # state means the state before taking action
    # action means the state take action to the new state
    # newState means the new state after taking action
    def updateQTable(self, state, action, newState, r: float):

        # mr means the max reward that the new state can get,
        # if the new state has been never got before, set 0
        mr = max(self.qtable[newState].values()) if newState in self.qtable else 0

        # calculate the q table's new value
        self.qtable[state][action] = (1 - self.alpha) * self.qtable[state][action] + self.alpha * (r + self.gamma * mr)

    # a step run in episode
    # state means current state
    # getActionSet means a function, args are state, return a set type variable means action set
    # getNewState means a function, args are state and action, return new state
    # getReward means a function, args are current state and action, return the reward that state take action
    # updateState means a function, args are new state, to make actual action and update state
    def stepRun(self, state, getActionSet, getNewState, getReward, updateState):

        actionSet = getActionSet(state)
        self.initState(state, actionSet)

        action = self.epsilon_greedy(state)  # choose an action
        newState = getNewState(state, action)  # get new state
        reward = getReward(state, action)  # get state:action reward
        self.updateQTable(state, action, newState, reward)  # update q table
        updateState(newState)  # update state
