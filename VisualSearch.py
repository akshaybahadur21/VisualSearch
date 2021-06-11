import argparse
import logging
import os
import sys

from src.classifier.Classifier import Classifier
from src.elasticsearch.ElasticSearch import ElasticSearch
from src.pretrained.PretrainedFactory import PretrainedFactory
from src.utils.utils import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from src.utils.utils import download_dataset


class VisualSearch:
    def __init__(self, yml_path):
        self.conf = get_config(yml_path)
        self.data_path = download_dataset(self.conf['dataset'])
        self.classifier = Classifier(self.conf['classifier'])
        self.pretrained = PretrainedFactory(self.conf['pretrained'])
        self.elasticsearch = ElasticSearch(self.conf['elasticsearch'])

    def search(self):
        self.classifier.classify(self.data_path)
        pretrained_model = self.pretrained.get_pretrained_model(self.conf['pretrained']['model.type'])
        filenames = self.elasticsearch.preprocess(self.data_path, pretrained_model, self.classifier)
        self.elasticsearch.perform_visual_search(filenames, pretrained_model, self.classifier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-yml', "--yml_path", required=True, help="Path to application.yaml", type=str)
    args = parser.parse_args()
    yml_path = args.yml_path
    if not os.path.isfile(yml_path):
        logger.error("The specified application.yaml %s path is not valid", yml_path)
        sys.exit()

    visualSearch = VisualSearch(yml_path)
    visualSearch.search()
