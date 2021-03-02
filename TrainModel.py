from DataProcessing import DataProcessing
import os


class TrainModel:
    directory = r'C:\Users\aminu\Documents\study materials (8th semester)\spl3\datasets'
    categories = ['00000', 'CMFD', 'IMFD']
    __dataProcessor = DataProcessing(os.path.join(directory, categories[0]))

    def __trainModel(self):
        self.__dataProcessor.getDataList()
        return "model"

    def getModel(self):
        return self.__trainModel()