import cv2


class FaceDetector:
    def __detectFace(self, inputImage):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        grayScaledImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayScaledImage, 1.1, 1)
        return faces

    def getDetectedFaceImage(self, inputImage):
        return self.__detectFace(inputImage)
