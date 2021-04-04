import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class DataProcessing:
    # subdirectories will be the categories like NO_MASK or IMFD or CMFD
    def __init__(self, directory, categories):
        self.__dataDirectory = directory
        self.__categories = categories

    __data = []
    __labels = []
    tempCount = 0

    def __parseImagesIntoDatalist(self, category):
        # joining the directory and category to get the path of the subdirectory
        directory = os.path.join(self.__dataDirectory, category)

        for subdirectory in os.listdir(directory):
            path = os.path.join(directory, subdirectory)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                print(img)
                # print(image.ndim)
                self.__data.append(image)
                self.__labels.append(category)
                self.tempCount += 1

    def getDataList(self):
        for category in self.__categories:
            self.__parseImagesIntoDatalist(category)
        print('total image read: ', self.tempCount)
        return self.__data

    def getLabelsList(self):
        return self.__labels
