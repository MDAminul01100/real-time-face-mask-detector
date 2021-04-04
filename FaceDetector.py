import cv2
import imutils
import numpy as np

class FaceDetector:
    __MINIMUM_CONFIDENCE = 0.5

    # detect face on an image using haarcascade
    # def __detectFace(self, inputImage):
    #     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #     grayScaledImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(grayScaledImage, 1.1, 1)
    #     return faces

    def __detectFace(self, frame):
        # load our serialized face detector model from disk
        prototxtPath = r"resources\face_detector\deploy.prototxt"
        weightsPath = r"resources\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # grab the frame dimensions and convert it to a blob and
        # resizing it to a fixed 300x300 pixels and then normalizing it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        faceNet.setInput(blob)
        detections = faceNet.forward()

        faceLocations = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < self.__MINIMUM_CONFIDENCE:
                continue
            # print(confidence)
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            face = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = face.astype("int")

            # making sure the bounding boxes around the faces
            # fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            faceLocations.append((startX, startY, endX, endY))

        return faceLocations


    def getDetectedFaces(self, inputFrame):
        return self.__detectFace(inputFrame)
