import logging
import os
import tarfile

import gdown
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def download_dataset(name):
    if name == "caltech_101":
        if os.path.exists("data/101_ObjectCategories"):
            return "data/101_ObjectCategories"
        url = "https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp"
        output = '101_ObjectCategories.tar.gz'
        gdown.download(url, output, quiet=False)
        file = tarfile.open("101_ObjectCategories.tar.gz")
        file.extractall('data')
        os.remove("101_ObjectCategories.tar.gz")
        file.close()
        return "data/101_ObjectCategories"

    else:
        logger.error("Unknown dataset name : %s", str(name))
        logger.exception('Got exception on main handler')


def plot_imgs(faces):
    images = []
    for face in faces:
        images.append(mpimg.imread(face))
    plt.figure(figsize=(20, 10))
    columns = 4
    for i, image in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)
    plt.show()
