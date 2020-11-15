from PyQt5 import QtCore, QtGui, QtWidgets
from tkinter import filedialog
from os import walk
import os
from TEST import model

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(300,100,999, 721)
        MainWindow.setStyleSheet("background-color: rgb(44, 49, 60);")
        MainWindow.show()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.LOAD_button = QtWidgets.QPushButton(self.centralwidget)
        self.LOAD_button.setGeometry(QtCore.QRect(20, 110, 301, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.LOAD_button.setFont(font)
        self.LOAD_button.setStyleSheet("QPushButton {\n"
"    color:rgb(220, 220, 220);\n"
"    border: 2px solid #555;\n"
"    border-radius: 20px;\n"
"    border-style:rgb(62, 67, 81);\n"
"    background:rgb(27, 29, 35);\n"
"    padding: 5px;\n"
"    }\n"
"\n"
"QPushButton:hover {\n"
"    color:rgb(27,29,35);\n"
"    background: rgb(220, 220, 220);\n"
"    }\n"
"")
        self.LOAD_button.setObjectName("LOAD_button")
        
        self.launch_button = QtWidgets.QPushButton(self.centralwidget)
        self.launch_button.setGeometry(QtCore.QRect(20, 190, 301, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setKerning(False)
        self.launch_button.setFont(font)
        self.launch_button.setStyleSheet("QPushButton {\n"
"    color:rgb(220, 220, 220);\n"
"    border: 2px solid #555;\n"
"    border-radius: 20px;\n"
"    border-style:rgb(62, 67, 81);\n"
"    background:rgb(45, 0, 68);\n"
"    padding: 5px;\n"
"    }\n"
"\n"
"QPushButton:hover {\n"
"    color:rgb(45, 0, 68);\n"
"    background: rgb(220, 220, 220);\n"
"    }\n"
"")
        self.launch_button.setObjectName("launch_button")
        
        self.label_title_bar_top = QtWidgets.QLabel(self.centralwidget)
        self.label_title_bar_top.setGeometry(QtCore.QRect(0, 0, 1001, 51))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_title_bar_top.setFont(font)
        self.label_title_bar_top.setStyleSheet("background-color: rgb(27, 29, 35) ;\n"
"color: rgb(230, 230, 230);")
        self.label_title_bar_top.setObjectName("label_title_bar_top")
        
        self.NO_stroke = QtWidgets.QLabel(self.centralwidget)
        self.NO_stroke.setGeometry(QtCore.QRect(100, 125, 1300, 521))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(26)
        font.setBold(True)
        font.setItalic(True) 
        font.setWeight(75)
        self.NO_stroke.setFont(font)
        self.NO_stroke.setAlignment(QtCore.Qt.AlignCenter)
        self.NO_stroke.setStyleSheet("color:rgb(221, 221, 221);\n"
"border: 2px ;\n"
"border-radius: 30px;\n"
"border-style:rgb(62, 67, 81);\n"
"background:rgb(0, 93, 0);\n"
"padding: 5px;")
        self.NO_stroke.setObjectName("NO_Stroke")
        self.NO_stroke.hide()
        
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(335, 70, 650, 650))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("C:/Users/Matthieu/Desktop/AVC_search.png"))
        self.image.setScaledContents(True)
        self.image.setObjectName("image")
        
        self.image2 = QtWidgets.QLabel(self.centralwidget)
        self.image2.setGeometry(QtCore.QRect(335, 70, 650, 650))
        self.image2.setText("")
        self.image2.setPixmap(QtGui.QPixmap("C:/Users/Matthieu/Desktop/AVC_search.png"))
        self.image2.setScaledContents(True)
        self.image2.setObjectName("image2")
        self.image2.hide()
        
        self.left = QtWidgets.QPushButton(self.centralwidget)
        self.left.setGeometry(QtCore.QRect(621, 650, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.left.setFont(font)
        self.left.setStyleSheet("color: rgb(222, 222, 222);")
        self.left.setObjectName("left")
        self.left.hide()
        
        self.right = QtWidgets.QPushButton(self.centralwidget)
        self.right.setGeometry(QtCore.QRect(672, 650, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.right.setFont(font)
        self.right.setStyleSheet("color: rgb(222, 222, 222);")
        self.right.setObjectName("right")
        self.right.hide()
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.LOAD_button.clicked.connect(self.choosefile)
        self.left.clicked.connect(self.goleft)
        self.right.clicked.connect(self.goright)
        self.launch_button.clicked.connect(self.analysis)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "StrokeSearch"))
        self.LOAD_button.setText(_translate("MainWindow", "Load Images"))
        self.launch_button.setText(_translate("MainWindow", "Launch the Analysis"))
        self.NO_stroke.setText(_translate("MainWindow", "No Stroke Detected"))
        self.label_title_bar_top.setText(_translate("MainWindow", "  STROKE search"))
        self.left.setText(_translate("MainWindow", "<"))
        self.right.setText(_translate("MainWindow", ">"))
        
    def choosefile(self):
        global onlyfiles
        
        filename = filedialog.askdirectory()
        print(filename)
        onlyfiles = []
        for filenames in walk(filename):
            onlyfiles.extend(filenames)
            break
        global y
        y=int(onlyfiles[2][0][:-4])
        print(onlyfiles[2][0][:-4])
        global load
        load=1
        self.image.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/'+str(y)+'.jpg'))
        self.left.show()
        self.right.show()
               
    def goright(self):
        global y
        global onlyfiles
        if os.path.isfile(onlyfiles[0]+'/'+str(y+1)+'.jpg'):
            y+=1
            self.image.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/'+str(y)+'.jpg'))
        else :
            y=int(onlyfiles[2][0][:-4])
            self.image.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/'+str(y)+'.jpg'))
        if stroke==1:
            self.image2.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/output_'+str(y)+'.jpg.png'))
            
    def goleft(self):
        global y
        global onlyfiles
        if os.path.isfile(onlyfiles[0]+'/'+str(y-1)+'.jpg'):
            y-=1
            self.image.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/'+str(y)+'.jpg'))
        else :
            y=int(onlyfiles[2][0][:-4])+50
            while not(os.path.isfile(onlyfiles[0]+'/'+str(y)+'.jpg')):
                y-=1
            self.image.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/'+str(y)+'.jpg'))
        if stroke==1:
            self.image2.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/output_'+str(y)+'.jpg.png'))
            
            
    def analysis(self):
        global onlyfiles
        print (onlyfiles)
        model(onlyfiles)
        global load
        if load==1:
            self.launch_button.hide()
            self.label_title_bar_top.setGeometry(QtCore.QRect(0, 0, 1500, 51))
            self.LOAD_button.hide()
            MainWindow.setGeometry(0,100,1500, 800)
     #       if not stroke :
      #           self.NO_stroke.show()
            
            global stroke
            stroke=1
            self.image2.show()
            self.image2.setGeometry(750, 70, 650, 650)
            self.image2.setPixmap(QtGui.QPixmap(onlyfiles[0]+'/output_'+str(y)+'.jpg.png'))
            self.image.setGeometry(50, 70, 650, 650)
            self.right.setStyleSheet("background-color: rgb(10, 10, 10) ;\n"
    "color: rgb(230, 230, 230);")
            self.right.setGeometry(1505-750, 735, 41, 41)
            self.left.setStyleSheet("background-color: rgb(10, 10, 10) ;\n"
    "color: rgb(230, 230, 230);")
            self.left.setGeometry(1454-750, 735, 41, 41)
                   
            

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

