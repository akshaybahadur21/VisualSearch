import json
import logging
import os
import random

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.utils.utils import plot_imgs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ElasticSearch:
    def __init__(self, conf):
        self.conf = conf
        self.client = Elasticsearch(http_compress=True,
                                    hosts=[{"host": self.conf["es.nodes"], "port": self.conf["es.port"]}])
        self.extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    def create_es_index(self, name, path):
        idx = name
        if self.client.indices.exists(idx):
            if self.conf["es.rewrite_index"]:
                logger.info("Index %s already exists in ElasticSearch", idx)
                logger.info("deleting '%s' index..." % idx)
                res = self.client.indices.delete(index=idx)
                logger.info(" response: '%s'" % res)
            else:
                return "Exists"
        with open(path) as f:
            request_body = json.load(f)
        logger.info("creating '%s' index..." % idx)
        res = self.client.indices.create(index=idx, body=request_body)
        logger.info(" response: '%s'" % res)
        return idx

    def get_file_list(self, root_dir):
        file_list = []
        counter = 1
        for root, directories, filenames in os.walk(root_dir):
            for filename in filenames:
                if any(ext in filename for ext in self.extensions):
                    file_list.append(os.path.join(root, filename))
                    counter += 1
        return file_list

    def index_batch(self, features, filenames, pred_list, INDEX_NAME):
        requests = []
        for i, doc in enumerate(features):
            request = {"_op_type": "index", "_index": INDEX_NAME, "filenames": filenames[i],
                       "image_vector": features[i], "class": pred_list[i]}
            requests.append(request)
        bulk(self.client, requests)

    def index_data(self, filenames, idx, pretrained_model, classifier_model):
        count = 0
        BATCH_SIZE = self.conf['es.batch_size']
        INDEX_NAME = idx
        for i in range(0, len(filenames), BATCH_SIZE):
            count += 1
            pred_list = classifier_model.get_predictions_list(filenames, i, BATCH_SIZE)
            feature_list = pretrained_model.get_feature_list(filenames, i, BATCH_SIZE)
            self.index_batch(feature_list, filenames, pred_list, INDEX_NAME)
            logger.info("Indexed {} documents.".format(count * BATCH_SIZE))

        self.client.indices.refresh(index=INDEX_NAME)
        logger.info("Done indexing.")

    def preprocess(self, data_path, pretrained_model, classifier_model):
        idx = self.create_es_index(name=self.conf["es.index"], path=self.conf["es.index_mapping_path"])
        filenames = sorted(self.get_file_list(data_path))
        if idx == "Exists":
            return filenames
        logger.info("Total number of files are %s", len(filenames))
        self.index_data(filenames, idx, pretrained_model, classifier_model)
        logger.info("Preprocessing Completed")
        return filenames

    def perform_visual_search(self, filenames, pretrained_model, classifier_model):
        random.seed(42)
        for i in range(11):
            random_image_index = random.randint(0, len(filenames))
            search_img = filenames[random_image_index]
            query_vector = pretrained_model.extract_features(search_img)
            classifier_pred = classifier_model.make_predictions(search_img)

            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['image_vector']) + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }

            response = self.client.search(
                index=self.conf["es.index"],
                body={
                    "size": self.conf["es.approx_search_size"],
                    "query": script_query,
                    "_source": {"includes": ["filenames", "class"]}
                }
            )

            img_list = []
            for hit in response["hits"]["hits"]:
                if len(img_list) == self.conf["es.search_size"]:
                    break
                if hit["_source"]["class"] != classifier_pred:
                    continue  # Not included in the resultset
                logger.info("id: {}, score: {}".format(hit["_id"], hit["_score"]))
                logger.info(hit["_source"]["filenames"])
                img_list.append(hit["_source"]["filenames"])
            plot_imgs(img_list)
