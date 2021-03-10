import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class DataProcessing:
    def __init__(self, directory=None):
        self.__dataDirectory = directory

    __data = []
    __labels = []

    def __parseImagesIntoDatalist(self):
        # self.__dataDirectory += r'\00000'

        for subdirectory in os.listdir(self.__dataDirectory):
            path = os.path.join(self.__dataDirectory, subdirectory)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                # print(img)
                self.__data.append(image)
                # self.__labels.append(category)

    def getDataList(self):
        self.__parseImagesIntoDatalist()
        return self.__data

    def getLabelsList(self):
        return self.__labels
