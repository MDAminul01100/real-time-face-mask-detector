import cv2
import numpy as np
from FaceDetector import FaceDetector
from TrainModel import TrainModel
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class FaceMaskDetector:
    __trainModel = TrainModel()
    __faceMaskNet = __trainModel.getModel()

    __faceDetector = FaceDetector()

    # Face detection and drawing rectangles
    def __detectFaceMask(self, frame):
        faceLocations = self.__faceDetector.getDetectedFaces(frame)
        faces = []
        if len(faceLocations) > 0:
            for (x1, y1, x2, y2) in faceLocations:
                face = frame[y1:y2, x1:x2]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                # cv2.imshow('blabla', face)
                # cv2.waitKey(1000) & 0xFF
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                # print((x1, y1, x2, y2))
            faces = np.array(faces, dtype="float32")
            predictions = self.__faceMaskNet.predict(faces, batch_size=32)
        return (faceLocations, predictions)

    def getMaskPredictedImage(self, frame):
        (faceLocations, predictions) = self.__detectFaceMask(frame)

        if len(faceLocations) > 0:
            for (rectangle, prediction) in zip(faceLocations, predictions):
                (startX, startY, endX, endY) = rectangle
                (correctlyMasked, incorrectlyMasked, no_mask) = prediction
                print(no_mask * 100, correctlyMasked * 100, incorrectlyMasked * 100)
                label = ''
                # B, G, R
                # Blue for Incorrectly masked
                # Green for Correctly masked
                # Red for No mask
                rectangleColor = (0, 0, 0)
                if max(no_mask, correctlyMasked, incorrectlyMasked) == no_mask:
                    label = "No Mask"
                    rectangleColor = (0, 0, 255)
                elif max(no_mask, correctlyMasked, incorrectlyMasked) == correctlyMasked:
                    label = "Correctly Masked"
                    rectangleColor = (0, 255, 0)
                else:
                    label = "Incorrectly Masked"
                    rectangleColor = (255, 0, 0)

                label = "{}: {:.2f}%".format(label, max(no_mask, correctlyMasked, incorrectlyMasked) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, rectangleColor, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)

        # cv2.imshow('Image test1.jpg', frame)
        # cv2.waitKey(30000) & 0xFF
        return frame


image = cv2.imread('resources/resources_for_testing/test2.jpg')
faceMaskDetector = FaceMaskDetector()
image = faceMaskDetector.getMaskPredictedImage(image)
cv2.imwrite('resources/resources_for_testing/Output Image test2.jpg', image)
key = cv2.waitKey(30000) & 0xFF
