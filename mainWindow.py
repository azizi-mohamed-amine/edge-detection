# -*- coding: utf-8 -*-
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import os
import math
from utility import *


class mainUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(mainUI, self).__init__()
        self.sel_boolean = False
        self.in_frame=None
        self.view_frame=None

        recognitionModelConfiguration = "model/circle.cfg"
        # recognitionModelWeights = "model/circle_7200.weights"
        recognitionModelWeights = "model/circle_10300.weights"
        self.recognition_net = cv2.dnn.readNetFromDarknet(recognitionModelConfiguration, recognitionModelWeights)

        self.confThreshold = 0.25  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold
        self.recogInputWidth = 416  # Width of network's input image
        self.recogInputHeight = 416  # Height of network's input image

    def setupUi(self, MainWindow):
        self.MainWindow = MainWindow
        Form.setObjectName("MainWindow")
        Form.resize(700, 544)
        self.input_groupBox = QtWidgets.QGroupBox(MainWindow)
        self.input_groupBox.setGeometry(QtCore.QRect(30, 15, 501, 511))
        self.input_groupBox.setAutoFillBackground(True)
        self.input_groupBox.setTitle("")
        self.input_groupBox.setObjectName("input_groupBox")

        self.open_pushButton = QtWidgets.QPushButton(self.input_groupBox)
        self.open_pushButton.setGeometry(QtCore.QRect(10, 20, 92, 36))
        self.open_pushButton.setObjectName("open_pushButton")
        self.open_pushButton.clicked.connect(self.open_pushButton_on_click)

        self.in_imglabel = QtWidgets.QLabel(self.input_groupBox)
        self.in_imglabel.setGeometry(QtCore.QRect(10, 60, 480, 440))
        self.in_imglabel.setObjectName("in_imglabel")
        self.in_imglabel.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.in_imglabel.setStyleSheet("background-color: rgb(206,206,206);\n")
        self.in_imglabel.setAlignment(QtCore.Qt.AlignCenter)

       

        self.recog_folder_pushButton = QtWidgets.QPushButton(MainWindow)
        self.recog_folder_pushButton.setGeometry(QtCore.QRect(560, 70, 111, 36))
        self.recog_folder_pushButton.setObjectName("recog_folder_pushButton")
        self.recog_folder_pushButton.clicked.connect(self.recog_folder_pushButton_on_click)

        self.recog_line_pushButton = QtWidgets.QPushButton(MainWindow)
        self.recog_line_pushButton.setGeometry(QtCore.QRect(560, 110, 111, 36))
        self.recog_line_pushButton.setObjectName("recog_line_pushButton")
        self.recog_line_pushButton.clicked.connect(self.recog_line_pushButton_on_click)

        

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.open_pushButton.setText(_translate("MainWindow", "Open"))
        self.in_imglabel.setText(_translate("MainWindow", ""))
       
        self.recog_folder_pushButton.setText(_translate("MainWindow", "Recognition Folder"))
        self.recog_line_pushButton.setText(_translate("MainWindow", "Recognition Line"))

    def open_pushButton_on_click(self):
        print("click open_pushButton")
        self.in_filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "",
                                                               "Image Files (*.jpg *.jpeg *.bmp *.png);;All Files (*)")
        if self.in_filename:
            self.in_frame = cv2.imread(self.in_filename)
            h, w, _ = self.in_frame.shape
            image = cv2.resize(self.in_frame, (470, (int)(470 * h / w)))
            if len(image.shape) < 3 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, byteValue = image.shape
            byteValue = byteValue * width
            qimage = QtGui.QImage(image, width, height, byteValue, QtGui.QImage.Format_RGB888)
            self.in_imglabel.setPixmap(QtGui.QPixmap.fromImage(qimage))
            self.view_frame=self.in_frame.copy()
            self.sel_boolean = True

            self.MainWindow.setWindowTitle("MainWindow "+ self.in_filename)

        else:
            if (self.sel_boolean):
                self.in_imglabel.clear()
            self.sel_boolean = False

            print("cancel open_pushButton")

    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def Analysis(self, image, outs):
        classIds =[]
        confidences =[]
        boxes =[]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        if(len(outs)==0):
            QtWidgets.QMessageBox.about(self, "Alert", "Cannot find plate from the image")
            return
        imageHeight, imageWidth, _ = image.shape

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * imageWidth)
                    center_y = int(detection[1] * imageHeight)
                    width = int(detection[2] * imageWidth)
                    height = int(detection[3] * imageHeight)
                    left = int(center_x - width / 2)
                    if (left < 0):
                        left = 0
                    top = int(center_y - height / 2)
                    if (top < 0):
                        top = 0
                    right = int(center_x + width / 2)
                    if (right > imageWidth):
                        right = imageWidth - 1
                    bottom = int(center_y + height / 2)
                    if (bottom < 0):
                        bottom = 0
                    if (bottom > imageHeight):
                        bottom = imageHeight - 1
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([top, left, bottom, right])




        # return boxes

        # if len(boxes) != 3 and len(boxes) != 2:
        #     print("List is empty")
        #     return 0
        for box in boxes:
            font_color = (0, 0, 255)
            box_color = (255, 0, 0)
            left, top, right, bottom = box[1], box[0], box[3], box[2]

            # Draw the bounding box
            cv2.rectangle(self.view_frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(img=self.view_frame, text="Circle", org=(left, top),thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255))

        ##################
        h, w, _=self.view_frame.shape
        show_frame=cv2.resize(self.view_frame, (600, (int)(600*h/w)))
        cv2.imshow("RecognitionResult", show_frame)
        cv2.waitKey(1000)
    def Analysis_box(self, image, outs):
        classIds =[]
        confidences =[]
        boxes =[]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        if(len(outs)==0):
            QtWidgets.QMessageBox.about(self, "Alert", "Cannot find plate from the image")
            return
        imageHeight, imageWidth, _ = image.shape

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * imageWidth)
                    center_y = int(detection[1] * imageHeight)
                    width = int(detection[2] * imageWidth)
                    height = int(detection[3] * imageHeight)
                    left = int(center_x - width / 2)
                    if (left < 0):
                        left = 0
                    top = int(center_y - height / 2)
                    if (top < 0):
                        top = 0
                    right = int(center_x + width / 2)
                    if (right > imageWidth):
                        right = imageWidth - 1
                    bottom = int(center_y + height / 2)
                    if (bottom < 0):
                        bottom = 0
                    if (bottom > imageHeight):
                        bottom = imageHeight - 1
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([top, left, bottom, right])

                    cv2.rectangle(self.view_frame, (left, top), (right, bottom), (255, 0, 0), 2)

                ##################
        h, w, _ = self.view_frame.shape
        # show_frame = cv2.resize(self.view_frame, (600, (int)(600 * h / w)))
        # cv2.imshow("Rect Image", show_frame)
        # cv2.waitKey(1)
        return boxes


    def Analysis1(self, image, outs, f_name):
        classIds =[]
        confidences =[]
        boxes =[]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        if(len(outs)==0):
            QtWidgets.QMessageBox.about(self, "Alert", "Cannot find plate from the image")
            return
        imageHeight, imageWidth, _ = image.shape

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * imageWidth)
                    center_y = int(detection[1] * imageHeight)
                    width = int(detection[2] * imageWidth)
                    height = int(detection[3] * imageHeight)
                    left = int(center_x - width / 2)
                    if (left < 0):
                        left = 0
                    top = int(center_y - height / 2)
                    if (top < 0):
                        top = 0
                    right = int(center_x + width / 2)
                    if (right > imageWidth):
                        right = imageWidth - 1
                    bottom = int(center_y + height / 2)
                    if (bottom < 0):
                        bottom = 0
                    if (bottom > imageHeight):
                        bottom = imageHeight - 1
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([top, left, bottom, right])


        nboxes = len(boxes)

        if nboxes != 3 and nboxes != 2:
            print("List is empty imgIdx = %s, nBox = %d" % (f_name, nboxes))

        for box in boxes:
            font_color = (0, 0, 255)
            box_color = (255, 0, 0)
            left, top, right, bottom = box[1], box[0], box[3], box[2]

            # Draw the bounding box
            cv2.rectangle(self.view_frame, (left, top), (right, bottom), box_color, 2)

        h, w, _=self.view_frame.shape
        show_frame=cv2.resize(self.view_frame, (600, (int)(600*h/w)))
        cv2.imshow("RecognitionResult", show_frame)
        cv2.waitKey(0)

    def recog_pushButton_on_click(self):
        print("clicked recog_pushButton")

        if (self.sel_boolean == False):
            QtWidgets.QMessageBox.about(self, "Alert", "Please select a image")
            return

        height, width, _ = self.in_frame.shape

        resized_image = cv2.resize(self.in_frame, (self.recogInputWidth, self.recogInputHeight))

        w = resized_image.shape[1]
        resize_scale = w/ width

        # cv.imshow("resized_image", resized_image)
        # cv.waitKey(100)
        blob = cv2.dnn.blobFromImage(resized_image, 1 / 255, (self.recogInputWidth, self.recogInputHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        self.recognition_net.setInput(blob, "data")

        # Runs the forward pass to get output of the output layers
        outs = self.recognition_net.forward(self.getOutputsNames(self.recognition_net))

        self.Analysis(self.in_frame, outs)

    def recog_folder_pushButton_on_click(self):
        img_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Image Folder")

        file_list=os.listdir(img_dir)
        cnt = 0
        for f_name in file_list:
            print(f_name)
            cnt += 1
            f_path =img_dir+"/"+f_name
            self.in_frame=cv2.imread(f_path)
            self.in_frame = cv2.resize(self.in_frame, (400, int(400 * self.in_frame.shape[0] / self.in_frame.shape[1])))
            self.view_frame = self.in_frame.copy()
            height, width, _ = self.in_frame.shape

            resized_image = cv2.resize(self.in_frame, (self.recogInputWidth, self.recogInputHeight))

            w = resized_image.shape[1]
            resize_scale = w / width

            blob = cv2.dnn.blobFromImage(resized_image, 1 / 255, (self.recogInputWidth, self.recogInputHeight),
                                         [0, 0, 0],
                                         1, crop=False)
            self.recognition_net.setInput(blob, "data")
            outs = self.recognition_net.forward(self.getOutputsNames(self.recognition_net))

            boxes = self.Analysis_box(self.in_frame, outs)
            nBoxes = len(boxes)
            if nBoxes == 2 or nBoxes == 3:
                p1, p2, vdir, r1, r2 = refineBoxes(boxes, height)
                basePt1, basePt2, partPt1, partPt2 = getLines(self.in_frame, p1, p2, vdir, r1, r2)

                copyed_frame = self.in_frame.copy()

                cv2.line(copyed_frame, basePt1, basePt2, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(copyed_frame, partPt1, partPt2, (255, 0, 0), 1, cv2.LINE_AA)

                show_frame = cv2.resize(copyed_frame, (600, (int)(600 * height / width)))
                cv2.imshow("RecognitionResult", show_frame)
                cv2.waitKey(0)

            else:
                print("nBoxes = %d" % (nBoxes))
    def recog_line_pushButton_on_click(self):
        if (self.sel_boolean == False):
            QtWidgets.QMessageBox.about(self, "Alert", "Please select a image")
            return

        self.in_frame = cv2.resize(self.in_frame, (400, int(400 * self.in_frame.shape[0] / self.in_frame.shape[1])))
        self.view_frame = self.in_frame.copy()

        height, width, _ = self.in_frame.shape

        resized_image = cv2.resize(self.in_frame, (self.recogInputWidth, self.recogInputHeight))

        w = resized_image.shape[1]
        resize_scale = w / width

        blob = cv2.dnn.blobFromImage(resized_image, 1 / 255, (self.recogInputWidth, self.recogInputHeight), [0, 0, 0],
                                     1, crop=False)
        self.recognition_net.setInput(blob, "data")
        outs = self.recognition_net.forward(self.getOutputsNames(self.recognition_net))

        boxes = self.Analysis_box(self.in_frame, outs)
        nBoxes = len(boxes)
        if nBoxes == 2 or nBoxes == 3:
            p1, p2, vdir, r1, r2 = refineBoxes(boxes, height)
            basePt1, basePt2, partPt1, partPt2 = getLines(self.in_frame, p1, p2, vdir, r1, r2)

            copyed_frame = self.in_frame.copy()

            cv2.line(copyed_frame, basePt1, basePt2, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(copyed_frame, partPt1, partPt2, (255, 0, 0), 1, cv2.LINE_AA)

            show_frame = cv2.resize(copyed_frame, (600, (int)(600 * height / width)))
            cv2.imshow("RecognitionResult", show_frame)
            cv2.waitKey(1000)

        else:
            print("nBoxes = %d" % (nBoxes))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = mainUI()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

