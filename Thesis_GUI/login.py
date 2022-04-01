# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import mysql.connector as mc
import res
from main_dashboard import Ui_MainWindow

connect = mc.connect(
        host='localhost',
        port='3306',
        user='root',
        password='1234',
        database='thesisproject'
    )

dbcursor = connect.cursor()
selectquery = "SELECT * from user_credentials"
dbcursor.execute(selectquery)
records = dbcursor.fetchall()

class Login_Window(object):

    def openwindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    def authentication(self):
        print(records)
        usertext = self.userName_TextBox.text()
        passtext = self.password_TextBox.text()
        for x in records:
            if (x[1] == usertext and x[2] == passtext):
                print("test")
                self.openwindow()
                Login_Window.hide()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(731, 541)
        MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(30, 20, 661, 500))
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(40, 30, 280, 430))
        self.label.setStyleSheet("border-image: url(:/images/login_bg.jpg);\n"
"background-color: rgba(0, 0, 0, 80);\n"
"border-top-left-radius: 50px;")
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(270, 30, 340, 430))
        self.label_3.setStyleSheet("background-color:rgba(255, 255, 255, 255);\n"
"border-bottom-right-radius:50px")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(90, 110, 141, 141))
        self.label_4.setStyleSheet("border-image: url(:/images/thesis_system_logo.png);\n"
"")
        self.label_4.setText("")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(400, 80, 100, 40))
        font = QtGui.QFont()
        font.setFamily("Aileron SemiBold")
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgba(0, 0, 0, 200);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.userName_TextBox = QtWidgets.QLineEdit(self.widget)
        self.userName_TextBox.setGeometry(QtCore.QRect(295, 150, 300, 40))
        font = QtGui.QFont()
        font.setFamily("Agrandir Thin")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.userName_TextBox.setFont(font)
        self.userName_TextBox.setStyleSheet("background-color: rgba(0, 0, 0, 0);\n"
"border:none;\n"
"border-bottom:2px solid rgba(46, 82, 101, 200);\n"
"color:rgba(0, 0, 0, 240);\n"
"padding-bottom:7px;")
        self.userName_TextBox.setAlignment(QtCore.Qt.AlignCenter)
        self.userName_TextBox.setObjectName("userName_TextBox")
        self.password_TextBox = QtWidgets.QLineEdit(self.widget)
        self.password_TextBox.setGeometry(QtCore.QRect(295, 215, 300, 40))
        font = QtGui.QFont()
        font.setFamily("Agrandir Thin")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.password_TextBox.setFont(font)
        self.password_TextBox.setStyleSheet("background-color: rgba(0, 0, 0, 0);\n"
"border:none;\n"
"border-bottom:2px solid rgba(46, 82, 101, 200);\n"
"color:rgba(0, 0, 0, 240);\n"
"padding-bottom:7px;")
        self.password_TextBox.setAlignment(QtCore.Qt.AlignCenter)
        self.password_TextBox.setObjectName("password_TextBox")
        self.login_btn = QtWidgets.QPushButton(self.widget)
        self.login_btn.setGeometry(QtCore.QRect(350, 320, 190, 40))
        font = QtGui.QFont()
        font.setFamily("Aileron Bold")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.login_btn.setFont(font)
        self.login_btn.setStyleSheet("QPushButton#pushButton{\n"
"background-color: rgba(0, 0, 0, 50);\n"
"    color:rgba(255, 255, 255, 210);\n"
"    border-radius:5px;\n"
"}\n"
"\n"
"QPushButton#pushButton:hover{\n"
"    background-color:rgba(0, 0, 0, 255);\n"
"}\n"
"\n"
"QPushButton#pushButton:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color:rgba(255, 50, 85, 1);\n"
"}\n"
"\n"
"QPushButton#pushButton_2, #pushButton_3, #pushButton_4, #pushButton_5{\n"
"    background-color: rgba(0, 0, 0, 0);\n"
"    color:rgba(85, 98, 112, 255);\n"
"}\n"
"\n"
"QPushButton#pushButton_2:hover, #pushButton_3:hover, #pushButton_4:hover, #pushButton_5:hover{\n"
"    color: rgba(131, 96, 53, 255);\n"
"}\n"
"\n"
"QPushButton#pushButton_2:pressed, #pushButton_3:pressed, #pushButton_4:pressed, #pushButton_5:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    color:rgba(91, 88, 53, 255);\n"
"}")
        self.login_btn.setObjectName("login_btn")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setGeometry(QtCore.QRect(60, 270, 231, 151))
        font = QtGui.QFont()
        font.setFamily("Aileron Bold")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_5.setWordWrap(True)
        self.label_5.setObjectName("label_5")
        self.frame = QtWidgets.QFrame(self.widget)
        self.frame.setGeometry(QtCore.QRect(550, 30, 61, 40))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setMinimumSize(QtCore.QSize(15, 15))
        self.pushButton_2.setMaximumSize(QtCore.QSize(15, 15))
        self.pushButton_2.setStyleSheet("border-image: url(:/images/minimize_icon.png);")
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setMinimumSize(QtCore.QSize(15, 15))
        self.pushButton_3.setMaximumSize(QtCore.QSize(15, 15))
        self.pushButton_3.setStyleSheet("border-image: url(:/images/close_icon.png);")
        self.pushButton_3.setText("")
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.pushButton_2.clicked.connect(lambda: MainWindow.showMinimized())
        self.pushButton_3.clicked.connect(lambda: MainWindow.close())
        self.login_btn.clicked.connect(lambda: self.authentication())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "LOGIN"))
        self.userName_TextBox.setPlaceholderText(_translate("MainWindow", "User Name"))
        self.password_TextBox.setPlaceholderText(_translate("MainWindow", "Password"))
        self.login_btn.setText(_translate("MainWindow", "LOGIN"))
        self.label_5.setText(_translate("MainWindow", "Face Mask Detection & Periocular Recognition"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Login_Window()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
