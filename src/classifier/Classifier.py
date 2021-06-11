import keras
import numpy as np
import tqdm
from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import Conv2D, Dense, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


class Classifier:
    def __init__(self, conf):
        self.conf = conf
        self.loaded_resnet = None
        self.resnet = ResNet50(include_top=False, weights='imagenet',
                               input_shape=(224, 224, 3))
        output = self.resnet.layers[-1].output
        output = keras.layers.Flatten()(output)
        self.resnet = Model(self.resnet.input, outputs=output)
        for layer in self.resnet.layers:
            layer.trainable = False

    def get_predictions_list(self, filenames, i, BATCH_SIZE):
        pred_list = []
        for j in tqdm.tqdm(range(i, min(i + BATCH_SIZE, len(filenames)))):
            pred_list.append(self.make_predictions(filenames[j]))
        return pred_list

    def classify(self, data_path):
        if not self.conf['train.classifier']:
            self.loaded_resnet = load_model(self.conf["model.path"])
            return
        model = Sequential()
        model.add(self.resnet)
        model.add(Dense(512, activation='relu', input_dim=(224, 224, 3)))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(49, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=self.conf["model.batch_size"],
            seed=42,
            class_mode='categorical',
            subset="training",
            shuffle=True)

        validation_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=self.conf["model.batch_size"],
            seed=42,
            class_mode='categorical',
            subset="validation",
            shuffle=True)

        history = model.fit_generator(train_generator,
                                      epochs=5,
                                      validation_data=validation_generator)
        model.save(self.conf["model.path"])
        self.loaded_resnet = load_model(self.conf["model.path"])

    def make_predictions(self, img_path):
        input_shape = (224, 224, 3)
        img = image.load_img(img_path, target_size=(
            input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        pred_probab = self.loaded_resnet.predict(preprocessed_img)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return pred_class