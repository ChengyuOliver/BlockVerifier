
import sys
import os
from PyQt5.QtCore import QTime
import PyQt5.QtWidgets
import Style
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from DataAnalysis import *
# from newMain import *
from Ui_newMain import *
import time
from matplotlib.figure import Figure
import random
 

import numpy as np
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
# from temp import *


class DataAnalysisWindow(QtWidgets.QWidget, Ui_DataAnalysisWindow):
    def __init__(self, parent=None):
        super(DataAnalysisWindow, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.DataFile.clicked.connect(self.DataFileClick)
        self.pushButtonReturn.clicked.connect(self.setReturnMain)

    def DataFileClick(self):
        dir_choose = QtWidgets.QFileDialog.getOpenFileName(
            self, "选取数据文件", os.getcwd())
        self.DataAddress.setPlainText(dir_choose[0])
        pass

    def setReturnMain(self):
        # bug
        self.parent.setCentralWidget(self.parent.stacked.currentWidget())


class NewMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(NewMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.stacked = QtWidgets.QStackedWidget()
        
        # menu connect
        self.actionMetroUI.triggered.connect(self.styleMetroUI)
        self.actionBlack.triggered.connect(self.styleBlack)
        self.actionDeepBlack.triggered.connect(self.styleDeepBlack)
        self.actionDefault.triggered.connect(self.styleDefault)

        # button connect
        self.pushButton_DataAnalysis.clicked.connect(
            self.setDataAnalysisWindow)
        self.pushButtonReturn.clicked.connect(self.selectMenuReturn)
        # self.StartAnalysis.clicked.connect(self.startDataAnalysis)
        self.pushButton_4.clicked.connect(self.showfigure3)
        self.pushButton_Platform.clicked.connect(self.showfigure1)
        self.pushButton_3.clicked.connect(self.showfigure2)
        self.pushButton.clicked.connect(self.run_clicked)
        self.timer = QTime()
        # widget replace
        # self.static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # self.addToolBar(NavigationToolbar(self.static_canvas, self.ReplaceWidget))
        # self.replaceLayout.addWidget(self.static_canvas)
        # self._static_ax = self.static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self.i=0
        # self._static_ax.plot(t, np.tan(t), ".")
        # self.static_canvas.figure.canvas.draw()
    def operate(self):
        for i in range(0,101):
            self.progressBar_2.setProperty("value", i)
            # time.sleep(random.randint(1,5)/10.0)
            # self.timer.start(100)
        # self.i+=1
        # self.progressBar_2.setProperty("value", self.i)
        pass
    def run_clicked(self):
        # for i in range(0,101):
            # self.progressBar_2.setProperty("value", i)
            # time.sleep(random.randint(1,5))
        # self.timer.start(10)
        # self.timer.timeout.connect(self.operate)
        for i in range(0,101):
            self.progressBar_2.setProperty("value", i)
            time.sleep(random.randint(1,5)/100.0)
        QMessageBox.about(self,"Congratulations","Successful Verification！")
            
        pass
    def showfigure2(self):
        self.stackedWidget.setCurrentIndex(2)
        pass    
    def showfigure1(self):
        self.stackedWidget.setCurrentIndex(1)
        pass
    def showfigure3(self):
        self.stackedWidget.setCurrentIndex(4)
        pass

    def styleMetroUI(self):
        self.setStyleSheet(Style.getMetroUIStyle())
        pass

    def styleBlack(self):
        self.setStyleSheet(Style.getBlackStyle())
        pass

    def styleDeepBlack(self):
        self.setStyleSheet(Style.getDeepBlackStyle())
        pass

    def styleDefault(self):
        self.setStyleSheet("")
        pass

    def setDataAnalysisWindow(self):
        self.stackedWidget.setCurrentIndex(3)
        pass

    def selectMenuReturn(self):
        self.stackedWidget.setCurrentIndex(0)

    def startDataAnalysis(self):
        if (self.DataFormat.currentIndex() == 0):
            # code

            self.progressBar.setProperty("value", 50)


            # data = data_prep.liar_prep()
            # datalabel = []
            # datasizes = []
            # for i in range(1, data[1].__len__()):
            #     datalabel.append(data[1][i][0])
            #     datasizes.append(data[1][i][1])
            # self._static_ax.pie(
            #     datasizes, labels=datalabel, autopct='%1.1f%%', shadow=False)
            # # self._static_ax.title(data[0])
            # self.static_canvas.figure.canvas.draw()

        elif (self.DataFormat.currentIndex() == 1):
            data = data_prep.signal_prep()
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myWin = NewMainWindow()
    myWin.show()
    sys.exit(app.exec_())
