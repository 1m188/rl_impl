import random


# Q-Learning
class Q_Learning:

    # alpha, gamma, epsilon are all >0 and <1
    def __init__(self, alpha: float, gamma: float, epsilon: float):

        # q table's value is also a dict, {action:reward} means
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
