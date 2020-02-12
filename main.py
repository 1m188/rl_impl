import sys
from enum import Enum
from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtGui import QPainter
from PySide2.QtCore import Qt, QRect, QTimer
from q_learning import Q_Learning


class Action(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


# this game means that in 4x4 grid network, the blue ball in the origin point
# move to the red ball position, if the blue ball catch the black rect, game over.
# the blue ball will learn how to choose a safe and quick path to red ball
class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.initData()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("RL")
        self.resize(400, 400)
        renderTimer = QTimer(self)
        renderTimer.timeout.connect(self.update)
        renderTimer.start(15)

    def initData(self):
        # pos limit
        self.posWidth = 4
        self.posHeight = 4

        # all pos
        self.agentPos = (0, 0)  # init state
        self.negRectPosList = ((1, 1), (1, 2), (2, 1))
        self.posElpPos = (2, 2)

        # q learning
        self.ql = Q_Learning(0.5, 0.5, 0.01)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.qlRun)
        self.timer.start(50)

    def paintEvent(self, event):
        # update all graphic
        interval = 5
        agent = QRect(self.width() / 4 * self.agentPos[0] + interval, self.height() / 4 * self.agentPos[1] + interval, self.width() / 4 - interval * 2, self.height() / 4 - interval * 2)
        negRectList = []
        for negRectPos in self.negRectPosList:
            negRectList.append(QRect(self.width() / 4 * negRectPos[0] + interval, self.height() / 4 * negRectPos[1] + interval, self.width() / 4 - 2 * interval, self.height() / 4 - 2 * interval))
        posElp = QRect(self.width() / 4 * self.posElpPos[0] + interval, self.height() / 4 * self.posElpPos[1] + interval, self.width() / 4 - 2 * interval, self.height() / 4 - 2 * interval)

        # start paint
        painter = QPainter(self)

        # draw line
        painter.setPen(Qt.black)
        for i in range(1, 4):
            painter.drawLine(i * self.width() / 4, 0, i * self.width() / 4, self.height())
            painter.drawLine(0, i * self.height() / 4, self.width(), i * self.height() / 4)

        # draw negative rect
        painter.setBrush(Qt.black)
        for negRect in negRectList:
            painter.drawRect(negRect)

        # anti-aliasing
        painter.setRenderHint(QPainter.Antialiasing, True)

        # draw positive ellipse
        painter.setPen(Qt.red)
        painter.setBrush(Qt.red)
        painter.drawEllipse(posElp)

        # draw agent
        painter.setPen(Qt.blue)
        painter.setBrush(Qt.blue)
        painter.drawEllipse(agent)

        super().paintEvent(event)

    def qlRun(self):
        # judge if game restart
        if self.agentPos == self.posElpPos or self.agentPos in self.negRectPosList:
            self.agentPos = (0, 0)
            return

        # action set
        actionSet = set()
        if self.agentPos[0] > 0:
            actionSet.add(Action.LEFT)
        if self.agentPos[0] < self.posWidth - 1:
            actionSet.add(Action.RIGHT)
        if self.agentPos[1] > 0:
            actionSet.add(Action.UP)
        if self.agentPos[1] < self.posHeight - 1:
            actionSet.add(Action.DOWN)

        # init state
        self.ql.initState(self.agentPos, actionSet)

        # get action with epsilon-greedy
        action = self.ql.epsilon_greedy(self.agentPos)

        # get new state with the action got just now
        newAgentPos = (self.agentPos[0] + action.value[0], self.agentPos[1] + action.value[1])

        # calculate the action's reward for current state
        reward = 0
        if newAgentPos == self.posElpPos:
            reward = 10
        elif newAgentPos in self.negRectPosList:
            reward = -10
        else:
            # here the reward is bigger if the distance is smaller
            reward = -((newAgentPos[0] - self.posElpPos[0])**2 + (newAgentPos[1] - self.posElpPos[1])**2)**0.5

        # update q table
        self.ql.updateQTable(self.agentPos, action, newAgentPos, reward)

        # make actual action, update state
        self.agentPos = newAgentPos


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    app.exec_()
