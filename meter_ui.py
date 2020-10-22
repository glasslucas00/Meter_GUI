# -*- coding: utf-8 -*-
# @Time    : 2020/5/2 15:32
# @Author  : luyekang
# @Email   : glasslucas00@gmail.com
# @File    : test_ui2.py
# @Software: PyCharm
from yy2 import Ui_Form
import meter
# import sys
# sys.setrecursionlimit(1000000)
import sys
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import csv
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer, QDateTime
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import os

class CamShow(QMainWindow,Ui_Form):
    def __init__(self, parent=None):
        super(CamShow, self).__init__(parent)
        self.setupUi(self)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.timesrecord=0
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.cwd = 'C:/Users/Admin/Desktop/FILE' # 获取当前程序文件位置
        self.filepath=''
    def slot_init(self):
        self.pushButtoncamer.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.pushButtonexit.clicked.connect(self.close)
        self.pushButtonbegin.clicked.connect(self.readrequest)
        self.pushButtonend.clicked.connect(self.endapp)
        self.pushButtoncontinue.clicked.connect(self.continueapp)
        self.pushButtonshow.clicked.connect(self.showrecord)
        self.pushButton_2.clicked.connect(self.showrecord)
        self.btn_chooseDir.clicked.connect(self.slot_btn_chooseDir)
        self.btn_chooseoDir.clicked.connect(self.slot_btn_chooseoDir)

    def slot_btn_chooseDir(self):
        dir_choose = QtWidgets.QFileDialog.getExistingDirectory(self,"选取文件夹",self.cwd)  # 起始路径

        if dir_choose == "":
            print("\n取消选择")
            return

        print("\n你选择的文件夹为:")
        self.filepath=dir_choose
        print(dir_choose)

    def slot_btn_chooseoDir(self):
        dir_choose = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件夹", self.cwd)  # 起始路径

        if dir_choose == "":
            print("\n取消选择")
            return

        print("\n你选择的文件夹为:")
        self.outpath = dir_choose
        print(dir_choose)
        # print(os.listdir(self.filepath))

    def continueapp(self):
        self.flag = 0
    def endapp(self):
        self.flag=1

    def showrecord(self):

        '''testdatafile.csv'''
        with open(self.name+'.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            header_row = next(reader)  # 跳过第一行
            Time, Dates, P_src, P_filter = [], [], [], []  # 声明存储列表
            for row in reader:
                # time = float(row[0])       # 先将字符串转换为数字
                # Time.append(time)          # 存储
                current_date = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')  # 将日期数据转换为datetime对象
                Dates.append(current_date)  # 存储日期
                p_src = float(row[1])  # 先将字符串转换为数字
                P_src.append(p_src)  # 存储
                # p_filter = float(row[3])   # 先将字符串转换为数字
                # P_filter.append(p_filter)  # 存储

        x = Dates  # 切片数组，删除标题
        y1 = P_src
        # y2 = P_filter
        fig = plt.figure(figsize=( 6,3)  , dpi=100)

        plt.ylim(0, 100)
        plt.plot(x, y1,'ro-', c='red', alpha=1,)
        # plt.plot(x, y2, c='black', alpha=1)

        plt.title('osc')
        plt.xlabel('time', fontsize=14)
        plt.ylabel('value', fontsize=14)
        plt.savefig("plot.jpg")
        # plt.show()

        image = cv2.imread('plot.jpg')
        show = cv2.resize(image, (600, 400))
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_showdata.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def creatfile(self,name):
        with open("%s.csv" % (name), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            # recordtime = datetime.datetime.now()
            writer.writerow(['time', 'value'])

    def adddata(self,datas, name):  # flist=[[time0,value0],[time1,value1]]
        with open("%s.csv" % (name), "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            # recordtime = datetime.datetime.now()
            for data in datas:
                writer.writerows([data])

    # creatfile('firat')
    # adddata([['ds', 'ds'], ['sd', 'sdd']], 'firat')
    def showTime(self):
        if self.timesrecord>0:
            if self.flag==1:
                pass
            else:
                # for img in os.listdir(self.filepath):
                #     print(img)
                # print('timereeco',self.timesrecord)
                self.record_num.setText(str(self.timesrecord) + '/' + str(self.alltime))
                # self.record_num.setNum(self.timesrecord)
                self.timesrecord = self.timesrecord - 1
                pross=self.alltime-self.timesrecord
                self.progressBar.setValue(pross)

                self.img=self.imgname[self.num]
                self.opoint = meter.markzero(self.img)
                angle = meter.decter(self.img,self.outpath,self.opoint)
                print(angle)
                recordtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.adddata([[str(recordtime),str(angle)]],self.name)

                pname, ptype = self.img.split('.')
                lastname = pname.split('/')[-1]

                print(self.outpath+'/'+lastname+'_fin.jpg')
                image = cv2.imread(self.outpath+'/'+lastname+'_fin.jpg')
                show = cv2.resize(image, (300, 300))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                self.label_img2.setPixmap(QtGui.QPixmap.fromImage(showImage))
                self.label_output.setNum(angle)
                self.num+=1
        else:
            self.timer.stop()

    def readrequest(self):
        self.flag = 0
        self.num=0
        hour=self.record_time.value()
        minute=self.record_minute.value()
        self.timesrecord=int((hour*60)/minute)
        self.alltime=self.timesrecord
        self.progressBar.setMaximum(self.alltime)
        self.record_num.setText(str(self.timesrecord)+'/'+str(self.alltime))
        self.timer = QTimer(self)
        self.imgname=[]
        for img in os.listdir(self.filepath):
            self.imgname.append(self.filepath+'/'+img)
        print(self.imgname)
        # self.opoint=[299,86]
        # self.progressBar.setValue(1)
        self.timer.timeout.connect(self.showTime)
        self.timer.start(1000*minute)
        self.name='datafile'
        self.creatfile(self.name)

        # self.record_num.setNum(50)
    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            # if msg==QtGui.QMessageBox.Cancel:
            #                     pass
            else:
                self.timer_camera.start(30)

                self.pushButtoncamer.setText(u'关闭')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.labelimg.clear()
            self.pushButtoncamer.setText(u'摄像头')

    def show_camera(self):
        flag, self.image = self.cap.read()
        # face = self.face_detect.align(self.image)
        # if face:
        #     pass
        show = cv2.resize(self.image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        # print(show.shape[1], show.shape[0])
        # show.shape[1] = 640, show.shape[0] = 480
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.labelimg.setPixmap(QtGui.QPixmap.fromImage(showImage))

        # self.x += 1
        # selheaf.label_move.move(self.x,100)

        # if self.x ==320:
        #     self.labelimg.raise_()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui=CamShow()
    ui.show()
    sys.exit(app.exec_())
