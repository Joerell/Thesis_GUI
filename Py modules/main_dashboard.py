# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_dashboard.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox, QFileDialog

#VGG-IMPORTS
from tensorflow import keras
from tensorflow.keras import Model
from keras.models import load_model

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import test
import os
import main_res


    
class Ui_MainWindow(object):
  
    def __init__(self):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        MainWindow.resize(1400, 800)
        MainWindow.setMinimumSize(QtCore.QSize(1400, 800))
        MainWindow.setMaximumSize(QtCore.QSize(1400, 800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("#sidenav{\n"
"background-color: rgb(255, 255, 255);\n"
"}")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mainbody = QtWidgets.QWidget(self.centralwidget)
        self.mainbody.setObjectName("mainbody")
        self.sidenav = QtWidgets.QWidget(self.mainbody)
        self.sidenav.setGeometry(QtCore.QRect(0, -1, 200, 781))
        self.sidenav.setObjectName("sidenav")
        self.side_nav_container = QtWidgets.QFrame(self.sidenav)
        self.side_nav_container.setGeometry(QtCore.QRect(0, -1, 203, 781))
        self.side_nav_container.setStyleSheet("")
        self.side_nav_container.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.side_nav_container.setFrameShadow(QtWidgets.QFrame.Raised)
        self.side_nav_container.setObjectName("side_nav_container")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.side_nav_container)
        self.verticalLayout.setContentsMargins(1, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.logo_name_frame = QtWidgets.QFrame(self.side_nav_container)
        self.logo_name_frame.setMinimumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.logo_name_frame.setFont(font)
        self.logo_name_frame.setStyleSheet("border-bottom: 1px solid;")
        self.logo_name_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.logo_name_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.logo_name_frame.setObjectName("logo_name_frame")
        self.name_lbl = QtWidgets.QLabel(self.logo_name_frame)
        self.name_lbl.setGeometry(QtCore.QRect(30, 10, 30, 30))
        self.name_lbl.setMinimumSize(QtCore.QSize(30, 30))
        self.name_lbl.setMaximumSize(QtCore.QSize(30, 30))
        self.name_lbl.setStyleSheet("border-image: url(:/images/thesis_system_logo_with_no_text.png);")
        self.name_lbl.setText("")
        self.name_lbl.setObjectName("name_lbl")
        self.logo_lbl = QtWidgets.QLabel(self.logo_name_frame)
        self.logo_lbl.setGeometry(QtCore.QRect(70, 20, 111, 16))
        font = QtGui.QFont()
        font.setFamily("PanAmTextCaps")
        font.setPointSize(13)
        self.logo_lbl.setFont(font)
        self.logo_lbl.setStyleSheet("border:none;")
        self.logo_lbl.setObjectName("logo_lbl")
        self.verticalLayout.addWidget(self.logo_name_frame)
        self.dashboard_btn = QtWidgets.QPushButton(self.side_nav_container)
        self.dashboard_btn.setEnabled(True)
        self.dashboard_btn.setMinimumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.dashboard_btn.setFont(font)
        self.dashboard_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.dashboard_btn.setStyleSheet("QPushButton#dashboard_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border: none;\n"
"\n"
"}\n"
"\n"
"QPushButton#dashboard_btn:hover{\n"
"background-color: rgb(241, 241, 241);\n"
"}\n"
"\n"
"QPushButton#dashboard_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/dashboard_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.dashboard_btn.setIcon(icon)
        self.dashboard_btn.setObjectName("dashboard_btn")
        self.verticalLayout.addWidget(self.dashboard_btn)
        self.surveillance_btn = QtWidgets.QPushButton(self.side_nav_container)
        self.surveillance_btn.setMinimumSize(QtCore.QSize(170, 100))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.surveillance_btn.setFont(font)
        self.surveillance_btn.setStyleSheet("QPushButton#surveillance_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border: none;\n"
"}\n"
"\n"
"QPushButton#surveillance_btn:hover{\n"
"background-color: rgb(241, 241, 241);\n"
"}\n"
"\n"
"QPushButton#surveillance_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/surveillance_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.surveillance_btn.setIcon(icon1)
        self.surveillance_btn.setObjectName("surveillance_btn")
        self.verticalLayout.addWidget(self.surveillance_btn)
        self.history_btn = QtWidgets.QPushButton(self.side_nav_container)
        self.history_btn.setMinimumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.history_btn.setFont(font)
        self.history_btn.setStyleSheet("QPushButton#history_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border: none;\n"
"}\n"
"\n"
"QPushButton#history_btn:hover{\n"
"background-color: rgb(241, 241, 241);\n"
"}\n"
"\n"
"QPushButton#history_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/surveillance_report.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.history_btn.setIcon(icon2)
        self.history_btn.setObjectName("history_btn")
        self.verticalLayout.addWidget(self.history_btn)
        self.vgg16_btn = QtWidgets.QPushButton(self.side_nav_container)
        self.vgg16_btn.setMinimumSize(QtCore.QSize(200, 100))
        self.vgg16_btn.setMaximumSize(QtCore.QSize(200, 100))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        self.vgg16_btn.setFont(font)
        self.vgg16_btn.setStyleSheet("QPushButton#vgg16_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border: none;\n"
"}\n"
"\n"
"QPushButton#vgg16_btn:hover{\n"
"background-color: rgb(241, 241, 241);\n"
"}\n"
"\n"
"QPushButton#vgg16_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        self.vgg16_btn.setObjectName("vgg16_btn")
        self.verticalLayout.addWidget(self.vgg16_btn)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.logout_btn = QtWidgets.QPushButton(self.side_nav_container)
        self.logout_btn.setMinimumSize(QtCore.QSize(200, 110))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        self.logout_btn.setFont(font)
        self.logout_btn.setStyleSheet("QPushButton#logout_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border: none;\n"
"}\n"
"\n"
"QPushButton#logout_btn:hover{\n"
"background-color: rgb(241, 241, 241);\n"
"}\n"
"\n"
"QPushButton#logout_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}\n"
"")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/logout_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.logout_btn.setIcon(icon3)
        self.logout_btn.setObjectName("logout_btn")
        self.verticalLayout.addWidget(self.logout_btn)
        self.stackedWidget = QtWidgets.QStackedWidget(self.mainbody)
        self.stackedWidget.setGeometry(QtCore.QRect(200, 39, 1170, 740))
        self.stackedWidget.setStyleSheet("background-color: rgb(241, 241, 241);\n"
"")
        self.stackedWidget.setObjectName("stackedWidget")
        self.dashboard_page = QtWidgets.QWidget()
        self.dashboard_page.setObjectName("dashboard_page")
        self.frame_3 = QtWidgets.QFrame(self.dashboard_page)
        self.frame_3.setGeometry(QtCore.QRect(20, 65, 800, 320))
        self.frame_3.setStyleSheet("border-radius:50px;\n"
"background-color: rgb(0, 0, 0,10);")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        self.frame_6.setStyleSheet("background:transparent;")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frame_5 = QtWidgets.QFrame(self.frame_6)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.eye_symbol_lbl = QtWidgets.QLabel(self.frame_5)
        self.eye_symbol_lbl.setGeometry(QtCore.QRect(10, 20, 131, 100))
        self.eye_symbol_lbl.setStyleSheet("image: url(:/images/daily_count.png);\n"
"border-radius:30px;\n"
"background-color: rgb(162, 162, 162, 50);\n"
"\n"
"")
        self.eye_symbol_lbl.setText("")
        self.eye_symbol_lbl.setObjectName("eye_symbol_lbl")
        self.detection_count_daily_lbl = QtWidgets.QLabel(self.frame_5)
        self.detection_count_daily_lbl.setGeometry(QtCore.QRect(10, 150, 150, 80))
        font = QtGui.QFont()
        font.setPointSize(50)
        font.setBold(True)
        font.setWeight(75)
        self.detection_count_daily_lbl.setFont(font)
        self.detection_count_daily_lbl.setText("")
        self.detection_count_daily_lbl.setObjectName("detection_count_daily_lbl")
        self.detection_count_lbl = QtWidgets.QLabel(self.frame_5)
        self.detection_count_lbl.setGeometry(QtCore.QRect(10, 120, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setItalic(True)
        self.detection_count_lbl.setFont(font)
        self.detection_count_lbl.setObjectName("detection_count_lbl")
        self.label_15 = QtWidgets.QLabel(self.frame_5)
        self.label_15.setGeometry(QtCore.QRect(10, 240, 80, 30))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setItalic(True)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_5.addWidget(self.frame_5)
        self.proper_frame = QtWidgets.QFrame(self.frame_6)
        self.proper_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.proper_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.proper_frame.setObjectName("proper_frame")
        self.label_14 = QtWidgets.QLabel(self.proper_frame)
        self.label_14.setGeometry(QtCore.QRect(10, 20, 130, 100))
        self.label_14.setMinimumSize(QtCore.QSize(130, 100))
        self.label_14.setMaximumSize(QtCore.QSize(100, 100))
        self.label_14.setStyleSheet("border-radius:30px;\n"
"background-color: rgb(162, 162, 162, 50);\n"
"image: url(:/images/proper_mask.png);")
        self.label_14.setText("")
        self.label_14.setObjectName("label_14")
        self.detection_count_lbl_3 = QtWidgets.QLabel(self.proper_frame)
        self.detection_count_lbl_3.setGeometry(QtCore.QRect(10, 120, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setItalic(True)
        self.detection_count_lbl_3.setFont(font)
        self.detection_count_lbl_3.setObjectName("detection_count_lbl_3")
        self.proper_count = QtWidgets.QLabel(self.proper_frame)
        self.proper_count.setGeometry(QtCore.QRect(10, 150, 150, 80))
        font = QtGui.QFont()
        font.setPointSize(50)
        font.setBold(True)
        font.setWeight(75)
        self.proper_count.setFont(font)
        self.proper_count.setText("")
        self.proper_count.setObjectName("proper_count")
        self.label_10 = QtWidgets.QLabel(self.proper_frame)
        self.label_10.setGeometry(QtCore.QRect(10, 240, 80, 30))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setItalic(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_5.addWidget(self.proper_frame)
        self.improper_frame = QtWidgets.QFrame(self.frame_6)
        self.improper_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.improper_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.improper_frame.setObjectName("improper_frame")
        self.label_16 = QtWidgets.QLabel(self.improper_frame)
        self.label_16.setGeometry(QtCore.QRect(10, 20, 130, 100))
        self.label_16.setMinimumSize(QtCore.QSize(130, 100))
        self.label_16.setMaximumSize(QtCore.QSize(100, 100))
        self.label_16.setStyleSheet("border-radius:30px;\n"
"image: url(:/images/improper_wearing.png);\n"
"background-color: rgb(162, 162, 162, 50);\n"
"")
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.detection_count_lbl_2 = QtWidgets.QLabel(self.improper_frame)
        self.detection_count_lbl_2.setGeometry(QtCore.QRect(10, 120, 221, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setItalic(True)
        self.detection_count_lbl_2.setFont(font)
        self.detection_count_lbl_2.setObjectName("detection_count_lbl_2")
        self.improper_count_lbl = QtWidgets.QLabel(self.improper_frame)
        self.improper_count_lbl.setGeometry(QtCore.QRect(10, 150, 150, 80))
        font = QtGui.QFont()
        font.setPointSize(50)
        font.setBold(True)
        font.setWeight(75)
        self.improper_count_lbl.setFont(font)
        self.improper_count_lbl.setText("")
        self.improper_count_lbl.setObjectName("improper_count_lbl")
        self.label_17 = QtWidgets.QLabel(self.improper_frame)
        self.label_17.setGeometry(QtCore.QRect(10, 240, 80, 30))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setItalic(True)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_5.addWidget(self.improper_frame)
        self.horizontalLayout_4.addWidget(self.frame_6)
        self.label_18 = QtWidgets.QLabel(self.dashboard_page)
        self.label_18.setGeometry(QtCore.QRect(20, 10, 201, 51))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(20)
        self.label_18.setFont(font)
        self.label_18.setObjectName("label_18")
        self.frame_8 = QtWidgets.QFrame(self.dashboard_page)
        self.frame_8.setGeometry(QtCore.QRect(850, 65, 270, 320))
        self.frame_8.setStyleSheet("border-radius:50px;\n"
"background-color: rgb(0, 0, 0,10);")
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.detection_count_lbl_5 = QtWidgets.QLabel(self.frame_8)
        self.detection_count_lbl_5.setGeometry(QtCore.QRect(30, 20, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setItalic(True)
        self.detection_count_lbl_5.setFont(font)
        self.detection_count_lbl_5.setObjectName("detection_count_lbl_5")
        self.label_2 = QtWidgets.QLabel(self.frame_8)
        self.label_2.setGeometry(QtCore.QRect(40, 70, 200, 200))
        self.label_2.setStyleSheet("\n"
"border-radius:10px;\n"
"border: 1px solid black;")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.stackedWidget.addWidget(self.dashboard_page)
        self.surveillance_page = QtWidgets.QWidget()
        self.surveillance_page.setObjectName("surveillance_page")
        self.recent_visitors = QtWidgets.QLabel(self.surveillance_page)
        self.recent_visitors.setGeometry(QtCore.QRect(750, 60, 300, 200))
        self.recent_visitors.setStyleSheet("background-color: rgb(163, 163, 163, 50);\n"
"border-radius: 10px;")
        self.recent_visitors.setText("")
        self.recent_visitors.setObjectName("recent_visitors")
        self.recent_visitor_surveillance_lbl = QtWidgets.QLabel(self.surveillance_page)
        self.recent_visitor_surveillance_lbl.setGeometry(QtCore.QRect(850, 20, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.recent_visitor_surveillance_lbl.setFont(font)
        self.recent_visitor_surveillance_lbl.setObjectName("recent_visitor_surveillance_lbl")
        self.camera_frame = QtWidgets.QLabel(self.surveillance_page)
        self.camera_frame.setGeometry(QtCore.QRect(30, 100, 600, 400))
        self.camera_frame.setStyleSheet("background-color: rgb(163, 163, 163, 50);\n"
"\n"
"border-radius: 10px;")
        self.camera_frame.setText("")
        self.camera_frame.setObjectName("camera_frame")
        self.widget_3 = QtWidgets.QWidget(self.surveillance_page)
        self.widget_3.setGeometry(QtCore.QRect(750, 290, 300, 111))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.widget_3.setFont(font)
        self.widget_3.setStyleSheet("background:transparent;")
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.visitor_date = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.visitor_date.setFont(font)
        self.visitor_date.setObjectName("visitor_date")
        self.verticalLayout_2.addWidget(self.visitor_date)
        self.visitor_class = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.visitor_class.setFont(font)
        self.visitor_class.setObjectName("visitor_class")
        self.verticalLayout_2.addWidget(self.visitor_class)
        self.visitor_name = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.visitor_name.setFont(font)
        self.visitor_name.setStyleSheet("")
        self.visitor_name.setObjectName("visitor_name")
        self.verticalLayout_2.addWidget(self.visitor_name)
        self.frame_2 = QtWidgets.QFrame(self.surveillance_page)
        self.frame_2.setGeometry(QtCore.QRect(710, 460, 381, 281))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame = QtWidgets.QFrame(self.frame_2)
        self.frame.setStyleSheet("background-color: none;")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.faceholder_1 = QtWidgets.QLabel(self.frame)
        self.faceholder_1.setMinimumSize(QtCore.QSize(100, 100))
        self.faceholder_1.setMaximumSize(QtCore.QSize(100, 100))
        self.faceholder_1.setStyleSheet("\n"
"border: 1px solid black;")
        self.faceholder_1.setText("")
        self.faceholder_1.setObjectName("faceholder_1")
        self.horizontalLayout_3.addWidget(self.faceholder_1)
        self.faceholder_2 = QtWidgets.QLabel(self.frame)
        self.faceholder_2.setMinimumSize(QtCore.QSize(100, 100))
        self.faceholder_2.setMaximumSize(QtCore.QSize(100, 100))
        self.faceholder_2.setStyleSheet("\n"
"border: 1px solid black;")
        self.faceholder_2.setText("")
        self.faceholder_2.setObjectName("faceholder_2")
        self.horizontalLayout_3.addWidget(self.faceholder_2)
        self.faceholder_3 = QtWidgets.QLabel(self.frame)
        self.faceholder_3.setMinimumSize(QtCore.QSize(100, 100))
        self.faceholder_3.setMaximumSize(QtCore.QSize(100, 100))
        self.faceholder_3.setStyleSheet("\n"
"border: 1px solid black;")
        self.faceholder_3.setText("")
        self.faceholder_3.setObjectName("faceholder_3")
        self.horizontalLayout_3.addWidget(self.faceholder_3)
        self.verticalLayout_3.addWidget(self.frame)
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.faceholder_4 = QtWidgets.QLabel(self.frame_4)
        self.faceholder_4.setMinimumSize(QtCore.QSize(100, 100))
        self.faceholder_4.setMaximumSize(QtCore.QSize(100, 100))
        self.faceholder_4.setStyleSheet("\n"
"border: 1px solid black;\n"
"")
        self.faceholder_4.setText("")
        self.faceholder_4.setObjectName("faceholder_4")
        self.horizontalLayout_6.addWidget(self.faceholder_4)
        self.faceholder_5 = QtWidgets.QLabel(self.frame_4)
        self.faceholder_5.setMinimumSize(QtCore.QSize(100, 100))
        self.faceholder_5.setMaximumSize(QtCore.QSize(100, 100))
        self.faceholder_5.setStyleSheet("\n"
"border: 1px solid black;")
        self.faceholder_5.setText("")
        self.faceholder_5.setObjectName("faceholder_5")
        self.horizontalLayout_6.addWidget(self.faceholder_5)
        self.faceholder_6 = QtWidgets.QLabel(self.frame_4)
        self.faceholder_6.setMinimumSize(QtCore.QSize(100, 100))
        self.faceholder_6.setMaximumSize(QtCore.QSize(100, 100))
        self.faceholder_6.setStyleSheet("border: 1px solid black;\n"
"")
        self.faceholder_6.setText("")
        self.faceholder_6.setObjectName("faceholder_6")
        self.horizontalLayout_6.addWidget(self.faceholder_6)
        self.verticalLayout_3.addWidget(self.frame_4)
        self.Recently_Detected_lbl = QtWidgets.QLabel(self.surveillance_page)
        self.Recently_Detected_lbl.setGeometry(QtCore.QRect(710, 430, 161, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.Recently_Detected_lbl.setFont(font)
        self.Recently_Detected_lbl.setObjectName("Recently_Detected_lbl")
        self.frame_7 = QtWidgets.QFrame(self.surveillance_page)
        self.frame_7.setGeometry(QtCore.QRect(140, 510, 391, 81))
        self.frame_7.setStyleSheet("background:transparent;\n"
"")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.resume_btn = QtWidgets.QPushButton(self.frame_7)
        self.resume_btn.setMinimumSize(QtCore.QSize(110, 40))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.resume_btn.setFont(font)
        self.resume_btn.setStyleSheet("QPushButton#resume_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border:none;\n"
"    border-radius: 5px;\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"QPushButton#resume_btn:hover{\n"
"    background-color: rgb(249, 249, 249);\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"\n"
"QPushButton#resume_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.resume_btn.setIcon(icon4)
        self.resume_btn.setIconSize(QtCore.QSize(15, 15))
        self.resume_btn.setObjectName("resume_btn")
        self.horizontalLayout_7.addWidget(self.resume_btn)
        self.pause_btn = QtWidgets.QPushButton(self.frame_7)
        self.pause_btn.setMinimumSize(QtCore.QSize(110, 40))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.pause_btn.setFont(font)
        self.pause_btn.setStyleSheet("QPushButton#pause_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border:none;\n"
"    border-radius: 5px;\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"QPushButton#pause_btn:hover{\n"
"    background-color: rgb(249, 249, 249);\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"\n"
"QPushButton#pause_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(220, 220, 220);\n"
"}")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/images/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pause_btn.setIcon(icon5)
        self.pause_btn.setIconSize(QtCore.QSize(15, 15))
        self.pause_btn.setObjectName("pause_btn")
        self.horizontalLayout_7.addWidget(self.pause_btn)
        self.end_btn = QtWidgets.QPushButton(self.frame_7)
        self.end_btn.setMinimumSize(QtCore.QSize(110, 40))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.end_btn.setFont(font)
        self.end_btn.setStyleSheet("QPushButton#end_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border:none;\n"
"    border-radius: 5px;\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"QPushButton#end_btn:hover{\n"
"    background-color: rgb(249, 249, 249);\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"\n"
"QPushButton#end_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        self.end_btn.setIcon(icon4)
        self.end_btn.setIconSize(QtCore.QSize(15, 15))
        self.end_btn.setObjectName("end_btn")
        self.horizontalLayout_7.addWidget(self.end_btn)
        self.label = QtWidgets.QLabel(self.surveillance_page)
        self.label.setGeometry(QtCore.QRect(20, 10, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.surveillance_page)
        self.comboBox.setGeometry(QtCore.QRect(30, 70, 601, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setItemText(1, "")
        self.stackedWidget.addWidget(self.surveillance_page)
        self.history_page = QtWidgets.QWidget()
        self.history_page.setObjectName("history_page")
        self.history_lbl = QtWidgets.QLabel(self.history_page)
        self.history_lbl.setGeometry(QtCore.QRect(20, 10, 161, 111))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(25)
        font.setBold(True)
        font.setWeight(75)
        self.history_lbl.setFont(font)
        self.history_lbl.setStyleSheet("background:transparent;\n"
"")
        self.history_lbl.setObjectName("history_lbl")
        self.tableWidget = QtWidgets.QTableWidget(self.history_page)
        self.tableWidget.setGeometry(QtCore.QRect(180, 30, 951, 621))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.tableWidget.setFont(font)
        self.tableWidget.setStyleSheet("")
        self.tableWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.tableWidget.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setGridStyle(QtCore.Qt.SolidLine)
        self.tableWidget.setWordWrap(True)
        self.tableWidget.setRowCount(15)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(5)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(4, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(180)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(50)
        self.tableWidget.verticalHeader().setDefaultSectionSize(43)
        self.edit_btn = QtWidgets.QPushButton(self.history_page)
        self.edit_btn.setGeometry(QtCore.QRect(40, 180, 100, 50))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.edit_btn.setFont(font)
        self.edit_btn.setStyleSheet("QPushButton#edit_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border:none;\n"
"    border-radius: 5px;\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"QPushButton#edit_btn:hover{\n"
"    \n"
"    background-color: rgb(249, 249, 249);\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"\n"
"QPushButton#edit_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        self.edit_btn.setObjectName("edit_btn")
        self.delete_btn = QtWidgets.QPushButton(self.history_page)
        self.delete_btn.setGeometry(QtCore.QRect(40, 280, 100, 50))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.delete_btn.setFont(font)
        self.delete_btn.setStyleSheet("QPushButton#delete_btn{\n"
"    background-color: rgb(255, 255, 255);\n"
"    color: rgb(0, 0, 0);\n"
"    border:none;\n"
"    border-radius: 5px;\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"QPushButton#delete_btn:hover{\n"
"    background-color: rgb(249, 249, 249);\n"
"    border-bottom: 5px solid rgb(223,223,225);\n"
"    border-left: 1px solid rgb(223,223,225);\n"
"    border-right: 1px solid rgb(223,223,225);\n"
"}\n"
"\n"
"QPushButton#delete_btn:pressed{\n"
"    padding-left:5px;\n"
"    padding-top:5px;\n"
"    background-color: rgb(182, 182, 182);\n"
"}")
        self.delete_btn.setObjectName("delete_btn")
        self.stackedWidget.addWidget(self.history_page)
        self.vgg16_page = QtWidgets.QWidget()
        self.vgg16_page.setObjectName("vgg16_page")
        self.frame_9 = QtWidgets.QFrame(self.vgg16_page)
        self.frame_9.setGeometry(QtCore.QRect(270, 90, 541, 451))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.frame_9)
        self.label_3.setStyleSheet("background-color: rgb(218, 218, 218);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_5.addWidget(self.label_3)
        self.classify_btn = QtWidgets.QPushButton(self.frame_9)
        self.classify_btn.setObjectName("classify_btn")
        self.verticalLayout_5.addWidget(self.classify_btn)
        self.alert_btn = QtWidgets.QPushButton(self.frame_9)
        self.alert_btn.setObjectName("alert_btn")
        self.verticalLayout_5.addWidget(self.alert_btn)
        self.frame_10 = QtWidgets.QFrame(self.vgg16_page)
        self.frame_10.setGeometry(QtCore.QRect(420, 30, 281, 40))
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_10)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.frame_10)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.classification_result_lbl = QtWidgets.QLabel(self.frame_10)
        self.classification_result_lbl.setText("")
        self.classification_result_lbl.setObjectName("classification_result_lbl")
        self.horizontalLayout_2.addWidget(self.classification_result_lbl)
        self.stackedWidget.addWidget(self.vgg16_page)
        self.top_margin = QtWidgets.QFrame(self.mainbody)
        self.top_margin.setGeometry(QtCore.QRect(200, 0, 1170, 40))
        self.top_margin.setStyleSheet("background-color: rgb(241, 241, 241);\n"
"border-bottom: 1px solid;\n"
"")
        self.top_margin.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.top_margin.setFrameShadow(QtWidgets.QFrame.Raised)
        self.top_margin.setObjectName("top_margin")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.top_margin)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_7 = QtWidgets.QLabel(self.top_margin)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("border:  none;")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_8.addWidget(self.label_7)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem1)
        self.minimize_btn = QtWidgets.QPushButton(self.top_margin)
        self.minimize_btn.setMinimumSize(QtCore.QSize(15, 15))
        self.minimize_btn.setMaximumSize(QtCore.QSize(15, 15))
        self.minimize_btn.setStyleSheet("border-image: url(:/images/minimize_icon.png);\n"
"border:none;")
        self.minimize_btn.setText("")
        self.minimize_btn.setObjectName("minimize_btn")
        self.horizontalLayout_8.addWidget(self.minimize_btn)
        self.close_btn = QtWidgets.QPushButton(self.top_margin)
        self.close_btn.setMinimumSize(QtCore.QSize(15, 15))
        self.close_btn.setMaximumSize(QtCore.QSize(15, 15))
        self.close_btn.setStyleSheet("border:none;\n"
"border-image: url(:/images/close_icon.png);")
        self.close_btn.setText("")
        self.close_btn.setObjectName("close_btn")
        self.horizontalLayout_8.addWidget(self.close_btn)
        self.horizontalLayout.addWidget(self.mainbody)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # StackedWidget click event
        self.dashboard_btn.clicked.connect(lambda: self.showDashboard())
        self.surveillance_btn.clicked.connect(lambda: self.showSurveillance())
        self.history_btn.clicked.connect(lambda: self.showHistory())
        self.vgg16_btn.clicked.connect(lambda: self.showVggg16())

        # Top menu bar function
        self.minimize_btn.clicked.connect(lambda: MainWindow.showMinimized())
        self.close_btn.clicked.connect(lambda: MainWindow.close())

        # Upload Button
        self.classify_btn.clicked.connect(lambda:self.fileUpload())
        self.alert_btn.clicked.connect(lambda:self.speechToText())

        # Logout Function
        self.logout_btn.clicked.connect(lambda: self.logout())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.logo_lbl.setText(_translate("MainWindow", "AGNITIO"))
        self.dashboard_btn.setText(_translate("MainWindow", "Dashboard"))
        self.surveillance_btn.setText(_translate("MainWindow", "Surveillance"))
        self.history_btn.setText(_translate("MainWindow", "History"))
        self.vgg16_btn.setText(_translate("MainWindow", "Vgg16"))
        self.logout_btn.setText(_translate("MainWindow", "Logout"))
        self.detection_count_lbl.setText(_translate("MainWindow", " Detection Count"))
        self.label_15.setText(_translate("MainWindow", "Per Day"))
        self.detection_count_lbl_3.setText(_translate("MainWindow", "Proper Wearing Count"))
        self.label_10.setText(_translate("MainWindow", "Per Day"))
        self.detection_count_lbl_2.setText(_translate("MainWindow", "Improper Wearing Count"))
        self.label_17.setText(_translate("MainWindow", "Per Day"))
        self.label_18.setText(_translate("MainWindow", "Dashboard"))
        self.detection_count_lbl_5.setText(_translate("MainWindow", "Recent Visitor"))
        self.recent_visitor_surveillance_lbl.setText(_translate("MainWindow", "Recent Visitor"))
        self.visitor_date.setText(_translate("MainWindow", "Date:"))
        self.visitor_class.setText(_translate("MainWindow", "Classification:"))
        self.visitor_name.setText(_translate("MainWindow", "Name:"))
        self.Recently_Detected_lbl.setText(_translate("MainWindow", "Recently Detected"))
        self.resume_btn.setText(_translate("MainWindow", "Resume"))
        self.pause_btn.setText(_translate("MainWindow", "Pause"))
        self.end_btn.setText(_translate("MainWindow", "End"))
        self.label.setText(_translate("MainWindow", "Surveillance"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Camera"))
        self.history_lbl.setText(_translate("MainWindow", "History"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Time"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Date"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Name"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Detection Wearing"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Periocular Extraction"))
        self.edit_btn.setText(_translate("MainWindow", "Edit"))
        self.delete_btn.setText(_translate("MainWindow", "Delete"))
        self.classify_btn.setText(_translate("MainWindow", "Classify"))
        self.alert_btn.setText(_translate("MainWindow", "Alert"))
        self.label_4.setText(_translate("MainWindow", "Classification:"))

    # Side Nav button function
    def showDashboard(self):
        self.stackedWidget.setCurrentWidget(self.dashboard_page)

    def showSurveillance(self):
        self.stackedWidget.setCurrentWidget(self.surveillance_page)

    def showHistory(self):
        self.stackedWidget.setCurrentWidget(self.history_page)

    def showVggg16(self):
        self.stackedWidget.setCurrentWidget(self.vgg16_page)
    
 # Logout Function
    def logout(self):
        msg = QMessageBox()
        msg.setWindowTitle("Logout")
        msg.setText("Are you sure you want to logout and exit?")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.buttonClicked.connect(self.openwindow)
        x = msg.exec_()

    def openwindow(self, i):
        if i.text() == 'OK':
                MainWindow.close()
        else:
                print("Stay")
    
    def fileUpload(self):
        global file_name  
        file_name, _ = QFileDialog.getOpenFileName(None, "Save File", "Downloads", "Images (*.png, *.jpeg, *.xmp, *.jpg)")
        print(file_name)    
        self.label_3.setScaledContents(True)
        self.label_3.setPixmap(QPixmap(file_name))

    def speechToText(self):
        #Model VGG-16 
        self.VGG16()
        self.classification_result_lbl.setText('The prediction is ' + prediction)
        
        #Speech To Text
        import pyttsx3
        engine = pyttsx3.init()
        # convert this text to speech
        text = "The picture is a " + prediction
        engine.say(text)
        # play the speech
        engine.runAndWait()
        
    def VGG16(self):
        global prediction
        
        new_model = load_model('model_weight\cat_vs_dog_model.h5')
        sample = file_name
        img_pred=image.load_img(sample,target_size=(224,224))
        img_pred=image.img_to_array(img_pred)
        img_pred=np.expand_dims(img_pred, axis=0)
        
        rslt= new_model.predict(img_pred)
        
        if rslt[0][0]>rslt[0][1]:
                prediction="cat"        
        else:
                prediction="dog"   
        print(prediction)
            
        
        
 
                        
                
                


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
