import sys
from enum import Enum
from PySide2.QtWidgets import QApplication, QWidget, QMenu, QAction
from PySide2.QtGui import QPainter
from PySide2.QtCore import Qt, QRectF, QTimer
from algorithm.q_learning import Q_Learning
from algorithm.sarsa import Sarsa


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
        self.rightClickMenu.show()  # i don't know if don't show before, the right mouse click menu show will be so slow, but if it show before, the menu will be quicker

        initBlueBallPosAction = QAction(self.rightClickMenu)
        initBlueBallPosAction.setText("initialize blue ball's position")
        initBlueBallPosAction.triggered.connect(self.initBlueBallPos)

        stopAction = QAction(self.rightClickMenu)
        stopAction.setText("stop")
        stopAction.triggered.connect(self.stopRL)

        qlStartAction = QAction(self.rightClickMenu)
        qlStartAction.setText("Q-Learning start")
        qlStartAction.triggered.connect(self.qlStart)

        sarsaStartAction = QAction(self.rightClickMenu)
        sarsaStartAction.setText("Sarsa start")
        sarsaStartAction.triggered.connect(self.sarsaStart)

        self.rightClickMenu.addAction(initBlueBallPosAction)
        self.rightClickMenu.addAction(stopAction)
        self.rightClickMenu.addAction(qlStartAction)
        self.rightClickMenu.addAction(sarsaStartAction)

        # render timer
        renderTimer = QTimer(self)
        renderTimer.timeout.connect(self.update)
        renderTimer.start(15)

    def initData(self):
        # pos limit
        self.posWidth = 4
        self.posHeight = 4

        # all pos
        self.agentInitPos = (0, 0)
        self.agentPos = None  # define and init state
        self.initBlueBallPos()
        self.negRectPosList = ((1, 1), (1, 2), (2, 1))
        self.posElpPos = (2, 2)

        self.rlObj = None
        self.stepRun = None
        self.timer = QTimer(self)

    def stopRL(self):
        if self.timer.isActive():
            self.timer.stop()
            self.timer.timeout.disconnect(self.stepRun)
            self.stepRun = None
            self.rlObj = None

    def startRL(self, rlObj, stepRun):
        if not self.timer.isActive():
            self.rlObj = rlObj
            self.stepRun = stepRun
            self.timer.timeout.connect(self.stepRun)
            self.timer.start(50)

    def initBlueBallPos(self):
        self.agentPos = self.agentInitPos

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
        widthInterval = self.width() / self.posWidth
        heightInterval = self.height() / self.posHeight

        # update all graphic
        interval = 5
        agent = QRectF(widthInterval * self.agentPos[0] + interval, heightInterval * self.agentPos[1] + interval, widthInterval - interval * 2, heightInterval - interval * 2)
        negRectList = []
        for negRectPos in self.negRectPosList:
            negRectList.append(QRectF(widthInterval * negRectPos[0] + interval, heightInterval * negRectPos[1] + interval, widthInterval - 2 * interval, heightInterval - 2 * interval))
        posElp = QRectF(widthInterval * self.posElpPos[0] + interval, heightInterval * self.posElpPos[1] + interval, widthInterval - 2 * interval, heightInterval - 2 * interval)

        # start paint
        painter = QPainter(self)

        # draw line
        painter.setPen(Qt.black)
        for i in range(1, self.posHeight):
            painter.drawLine(0, i * heightInterval, self.width(), i * heightInterval)
        for i in range(1, self.posWidth):
            painter.drawLine(i * widthInterval, 0, i * widthInterval, self.height())

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
        self.initBlueBallPos()  # may be the last training result that the pos in the normal pos
        self.startRL(Q_Learning(0.5, 0.5, 0.01), self.qlStepRun)

    def qlStepRun(self):
        # judge if game restart
        if self.agentPos == self.posElpPos or self.agentPos in self.negRectPosList:
            self.initBlueBallPos()
            return
        self.rlObj.stepRun(self.agentPos, self.getActionSet, self.getNewState, self.getReward, self.updateState)

    def sarsaStart(self):
        rlObj = Sarsa(0.5, 0.5, 0.01)
        self.initBlueBallPos()
        rlObj.initAction(self.agentPos, self.getActionSet)
        self.startRL(rlObj, self.sarsaStepRun)

    def sarsaStepRun(self):
        if self.agentPos == self.posElpPos or self.agentPos in self.negRectPosList:
            self.initBlueBallPos()
            self.rlObj.initAction(self.agentPos, self.getActionSet)
            return
        self.rlObj.stepRun(self.agentPos, self.getActionSet, self.getNewState, self.getReward, self.updateState)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    app.exec_()
