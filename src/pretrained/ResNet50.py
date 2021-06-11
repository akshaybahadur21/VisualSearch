import numpy as np
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing import image
from numpy.linalg import norm
import tqdm


class ResNet:
    def __init__(self, conf):
        self.conf = conf
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='max',
                              input_shape=(224, 224, 3))

    def get_feature_list(self, filenames, i, BATCH_SIZE):
        feature_list = []
        for j in tqdm.tqdm(range(i, min(i + BATCH_SIZE, len(filenames)))):
            feature_list.append(self.extract_features(filenames[j]))
        return feature_list

    def extract_features(self, img_path):
        input_shape = (224, 224, 3)
        img = image.load_img(img_path, target_size=(
            input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = self.model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)
        return normalized_features
