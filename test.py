from DataProcessing import DataProcessing
from TrainModel import TrainModel
from FaceDetector import FaceDetector
import cv2
### train the model is below
# directory = r'C:\Users\aminu\Documents\study materials (8th semester)\spl3\datasets'
# dp = DataProcessing(directory)
# trainModel = TrainModel()
# trainModel.getModel()


### Face detection
img = cv2.imread('resources/test1.jpg')
faceDetector = FaceDetector()
faces = faceDetector.getDetectedFaceImage(img)
for (x, y, width, height) in faces:
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
cv2.imshow('Face Detected Image', img)
cv2.waitKey()


