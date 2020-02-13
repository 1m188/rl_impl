import random


# Sarsa
# sarsa is very similar to q learning, sarsa is also use q table to learn,
# the main difference is that q learning's action about evaluation and making
# choice are different, but sarsa's are the same. Q learning is off-poilcy,
# sarsa is on-poilcy. The q table update ways between them are also different.
# Initialize Q arbitrarily
# Repeat (for each episode):
#     Initialize s
#     Choose a from s using policy derived from Q(ε-greedy)
#     Repeat (for each step of episode):
#         Take action a, observe r, s'
#         Choose a' from s' using policy derived from Q(ε-greedy)
#         Q(s,a) = (1 - α) * Q(s,a) + α * [r + γ * Q(s',a')]
#         s = s' ; a = a'
#     until s is terminal
class Sarsa:

    # alpha, gamma, epsilon are all >0 and <1
    def __init__(self, alpha: float, gamma: float, epsilon: float):

        # self.qtable's values are also dicts, {action:reward} means
        # the action can take in this state and the action's reward
        self.qtable = {}  # q table
        self.alpha = alpha  # learn rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # epsilon-greedy

        self.action = None  # the action that sarsa choose will be done

    # initialize q table with the current state, if the state has been never got
    # before, set the reward of all actions about the state are 0
    # actionSet is the action set that the state can take
    def initQTable(self, state, actionSet: set):
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
    # newAction means the new state take action
    def updateQTable(self, state, action, newState, newAction, r: float):

        # nr means the reward new state take new action can get
        nr = self.qtable[newState][newAction]

        # calculate the q table's new value
        self.qtable[state][action] = (1 - self.alpha) * self.qtable[state][action] + self.alpha * (r + self.gamma * nr)

    # every episode's first step is choose an action with epsilon-greedy
    # state is current state
    # getActionSet is a function that get a set type variable from state
    def initAction(self, state, getActionSet):
        self.initQTable(state, getActionSet(state))
        self.action = self.epsilon_greedy(state)

    # a step run in episode
    # it's similar to q learning, the difference is the get new state through action that
    # has been chosen last step, and update q table use the reward that new state take
    # new action, and in the end update state and action preparing for next step
    def stepRun(self, state, getActionSet, getNewState, getReward, updateState):

        self.initQTable(state, getActionSet(state))

        # get new state and state:action reward
        newState = getNewState(state, self.action)
        reward = getReward(state, self.action)

        self.initQTable(newState, getActionSet(newState))

        # get new state take new action
        newAction = self.epsilon_greedy(newState)

        # update q table
        self.updateQTable(state, self.action, newState, newAction, reward)  # update q table

        # update state and action
        updateState(newState)
        self.action = newAction
