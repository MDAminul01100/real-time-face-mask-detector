from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras.models import load_model
from DataProcessing import DataProcessing
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import os


class TrainModel:
    __datasetDirectory = r'C:\Users\aminu\Documents\study materials (8th semester)\spl3\temp_dataset'

    # Correctly masked face Dataset - CMFD
    # Incorrectly masked face dataset - IMFD
    __categories = ['NO_MASK', 'CMFD', 'IMFD']
    __dataProcessor = DataProcessing(__datasetDirectory, __categories)

    # initialising initial learning rate, epochs and batch size
    __INIT_LR = 1e-4
    __EPOCHS = 20
    __BS = 32

    # constructing the training image generator for data augmentation
    __augmentation = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    def __oneHotEncoding(self, labels, lb):
        labels = lb.fit_transform(labels)
        # labels = to_categorical(labels)
        return labels

    def __constructHeadModel(self, baseModel):
        # construct the head of the model that will be placed on top of the the base model
        # by pooling, flattening and using dropout to avoid over-fitting of the model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        numberOfLayers = len(self.__categories)
        headModel = Dense(numberOfLayers, activation="softmax")(headModel)
        return headModel

    def __plotTrainingLossAndAccuracy(self, modelHistory):

        print(modelHistory.history.keys())

        N = self.__EPOCHS
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), modelHistory.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), modelHistory.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), modelHistory.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), modelHistory.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("resources/plot.png")

    def __trainModel(self):
        print('Processing data... ... ...')
        data = self.__dataProcessor.getDataList()
        labels = self.__dataProcessor.getLabelsList()

        # performing one-hot encoding on the labels
        lb = LabelBinarizer()
        labels = self.__oneHotEncoding(labels, lb)

        # making both the data and the labels to numpy arrays
        data = np.array(data, dtype="float32")
        labels = np.array(labels)

        # (trainX, testX, trainY, testY) = train_test_split(data, labels,
        #                                                   test_size=0.30, stratify=labels, random_state=42)

        # loading the MobileNetV2 and making sure the head fully connected layer sets are left off
        baseModel = MobileNetV2(weights="imagenet", include_top=False,
                                input_tensor=Input(shape=(224, 224, 3)))

        headModel = self.__constructHeadModel(baseModel)
        mainModel = Model(inputs=baseModel.input, outputs=headModel)

        # stopping update of the layers in the first training process
        for layer in baseModel.layers:
            layer.trainable = False

        optimizer = Adam(lr=self.__INIT_LR, decay=self.__INIT_LR / self.__EPOCHS)
        mainModel.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # training the head of the network
        print("[INFO] training head...")
        modelHistory = mainModel.fit(
            data,
            labels,
            validation_split=0.3,
            batch_size=self.__BS,
            epochs=self.__EPOCHS,
            shuffle=True
        )

        # # make predictions on the testing set
        # indexOfPredictedLabels = mainModel.predict(testX, batch_size=self.__BS)
        #
        # # for each image in the testing set we need to find the index of the
        # # label with corresponding largest predicted probability
        # indexOfPredictedLabels = np.argmax(indexOfPredictedLabels, axis=1)
        #
        # # show a good formatted classification report
        # print(classification_report(testY.argmax(axis=1), indexOfPredictedLabels,
        #                             target_names=lb.classes_))

        # saving the model to disk
        modelDirectory = r'resources\smart_face_mask_detector.model'
        mainModel.save(modelDirectory, save_format="h5")

        # plot the training loss and accuracy
        self.__plotTrainingLossAndAccuracy(modelHistory)

        return mainModel

    def getModel(self):
        projectRootDirectory = Path(__file__).parent
        modelDirectory = os.path.join(projectRootDirectory, 'resources')

        # checking if the root directory already contains the trained model, if not, then train a new model
        for file in os.listdir(modelDirectory):
            if file.endswith('.model'):
                print('model found in the directory... ...')
                return load_model('resources/smart_face_mask_detector.model')
        else:
            print('Model not found in the resource directory. Training a new model... ...')
            return self.__trainModel()
