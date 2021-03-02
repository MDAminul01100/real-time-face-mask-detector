from DataProcessing import DataProcessing
from TrainModel import TrainModel

directory = r'C:\Users\aminu\Documents\study materials (8th semester)\spl3\datasets'
dp = DataProcessing(directory)
trainModel = TrainModel()
trainModel.getModel()


