import sys
from enum import Enum
from PySide2.QtWidgets import QApplication, QWidget, QMenu, QAction
from PySide2.QtGui import QPainter
from PySide2.QtCore import Qt, QRectF, QTimer
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
        self.setWindowTitle("FindBall")
        self.resize(400, 400)

        # mouse right click menu
        self.rightClickMenu = QMenu(self)
        self.rightClickMenu.show()

        initBlueBallPosAction = QAction(self.rightClickMenu)
        initBlueBallPosAction.setText("initialize blue ball's position")
        initBlueBallPosAction.triggered.connect(self.initBlueBallPos)

        stopAction = QAction(self.rightClickMenu)
        stopAction.setText("stop")
        stopAction.triggered.connect(self.stopRL)

        qlStartAction = QAction(self.rightClickMenu)
        qlStartAction.setText("Q-Learning start")
        qlStartAction.triggered.connect(self.qlStart)

        self.rightClickMenu.addAction(initBlueBallPosAction)
        self.rightClickMenu.addAction(stopAction)
        self.rightClickMenu.addAction(qlStartAction)

        # render timer
        renderTimer = QTimer(self)
        renderTimer.timeout.connect(self.update)
        renderTimer.start(15)

    def initData(self):
        # pos limit
        self.posWidth = 4
        self.posHeight = 4

        # all pos
        self.agentPos = (0, 0)  # define and init state
        self.negRectPosList = ((1, 1), (1, 2), (2, 1))
        self.posElpPos = (2, 2)

        self.rlObj = None
        self.algorithm = None
        self.timer = QTimer(self)
        self.timer.setInterval(50)

    def stopRL(self):
        if self.timer.isActive() and self.algorithm:
            self.timer.stop()
            self.timer.timeout.disconnect(self.algorithm)
            self.algorithm = None
            self.rlObj = None

    def startRL(self, rlObj, algorithm):
        if not self.timer.isActive() and not self.algorithm:
            self.initBlueBallPos()
            self.rlObj = rlObj
            self.algorithm = algorithm
            self.timer.timeout.connect(self.algorithm)
            self.timer.start()

    def initBlueBallPos(self):
        self.agentPos = (0, 0)

    def getActionSet(self, agentPos: tuple) -> set:
        actionSet = set()
        if agentPos[0] > 0:
            actionSet.add(Action.LEFT)
        if agentPos[0] < self.posWidth - 1:
            actionSet.add(Action.RIGHT)
        if agentPos[1] > 0:
            actionSet.add(Action.UP)
        if agentPos[1] < self.posHeight - 1:
            actionSet.add(Action.DOWN)
        return actionSet

    def getNewState(self, agentPos: tuple, action: Action) -> tuple:
        return (agentPos[0] + action.value[0], agentPos[1] + action.value[1])

    def getReward(self, agentPos: tuple, action: Action) -> float:
        newState = self.getNewState(agentPos, action)
        reward = 0
        if newState == self.posElpPos:
            reward = 10
        elif newState in self.negRectPosList:
            reward = -10
        else:
            reward = -((newState[0] - self.posElpPos[0])**2 + (newState[1] - self.posElpPos[1])**2)**0.5
        return reward

    def updateState(self, newState: tuple):
        self.agentPos = newState

    def contextMenuEvent(self, event):
        self.rightClickMenu.exec_(event.globalPos())
        super().contextMenuEvent(event)

    def paintEvent(self, event):
        # update all graphic
        interval = 5
        agent = QRectF(self.width() / 4 * self.agentPos[0] + interval, self.height() / 4 * self.agentPos[1] + interval, self.width() / 4 - interval * 2, self.height() / 4 - interval * 2)
        negRectList = []
        for negRectPos in self.negRectPosList:
            negRectList.append(QRectF(self.width() / 4 * negRectPos[0] + interval, self.height() / 4 * negRectPos[1] + interval, self.width() / 4 - 2 * interval, self.height() / 4 - 2 * interval))
        posElp = QRectF(self.width() / 4 * self.posElpPos[0] + interval, self.height() / 4 * self.posElpPos[1] + interval, self.width() / 4 - 2 * interval, self.height() / 4 - 2 * interval)

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

    def qlStart(self):
        self.startRL(Q_Learning(0.5, 0.5, 0.01), self.qlRun)

    def qlRun(self):
        # judge if game restart
        if self.agentPos == self.posElpPos or self.agentPos in self.negRectPosList:
            self.initBlueBallPos()
            return
        self.rlObj.run(self.agentPos, self.getActionSet, self.getNewState, self.getReward, self.updateState)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    app.exec_()
