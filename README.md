# VisualSearch 🖼️ 🔍

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Emojinator/blob/master/LICENSE.md)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

A Hierarchical-Learning Reverse Image Search Engine pipeline for humans. 

Hierarchical modules:
- Classification
- Transfer Learning based feature generator

## Code Requirements 🦄
You can install Conda for python which resolves all the dependencies for machine learning.

`pip install -r requirements.txt`

## Description 🌈

Reverse image search (or as it is more technically known, instance retrieval) enables developers and researchers to build scenarios beyond simple keyword search. From discovering visually similar objects on Pinterest to recommending similar songs on Spotify to camera-based product search on Amazon, a similar class of technology under the hood is used. This is inspired by the following modules 
- [Google Lens](https://lens.google/) 
- [Amazon Camera Search](https://www.amazon.com/b?ie=UTF8&node=17387598011)


## Modules 💫
- Image Similarity
- Feature Extraction
- Similarity Search

## Dataset 🗃️
Currently, I have used the [Caltech 101 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/). In this dataset, we have pictures of objects belonging to 101 categories. About 40 to 800 images per category and most categories have about 50 images. However, the learnings from this can be extrapolated for domain specific datasets.

## Search Engine 🕊️
[Elasticsearch](https://www.elastic.co/) has been used as a search engine. Elasticsearch is a search engine based on the Lucene library. It provides a distributed, multitenant-capable full-text search engine with an HTTP web interface and schema-free JSON documents. Specifically, this pipeline uses [`dense_vactor`](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html) field in elasticsearch. A `dense_vector` field stores dense vectors of float values. 

## File Organization 🗄️

```shell
├── VisualSearch (Current Directory)
    ├── data : Data folder
        ├── a
            ├── 1.png
            ├── 2.png
            .
            .
        ├── b
            .
            .
        └── z
    ├── models : Trained classifier model
        └── classifier.h5 : classifier model
    ├── resources : resources folder
    ├── src : Source code
        ├── classifier
            └── Classifier.py
        ├── elasticsearch
            └── ElasticSearch.py
        ├── pretrained
            ├── PretrainedFactory.py
            └── ResNet50.py
        └── utils
            └── utils.py
    ├── VisualSearch.py : Driving class
    ├── application.yaml : yaml properties
    ├── LICENSE
    ├── requirements.txt
    └── readme.md
        
```

## Python  Implementation 👨‍🔬

1) Classifier : Training on pretrained - ResNet50 for classifying
2) Feature Extractor : Pretrained ResNet50 for feature extraction
3) Search Engine : Elasticsearch as a search engine

If you face any problem, kindly raise an issue

## Setup 🖥️

1) The pipeline is setup to be a one-click program.
2) Configure the `application.yaml` file
3) Run `VisualSearch.py` to run the application.


## Execution 🐉

```
python3 VisualSearch.py -yml application.yaml
```

## Results 📊
<img src="https://github.com/akshaybahadur21/VisualSearch/blob/main/resources/bike.png">
<img src="https://github.com/akshaybahadur21/VisualSearch/blob/main/resources/plane.png">
<img src="https://github.com/akshaybahadur21/VisualSearch/blob/main/resources/tiger.png">
<img src="https://github.com/akshaybahadur21/VisualSearch/blob/main/resources/dolphin.png">
<img src="https://github.com/akshaybahadur21/VisualSearch/blob/main/resources/face.png">

## References: 🔱
 
 - [Practical Deep Learning for Cloud, Mobile, and Edge by Anirudh Koul, Siddha Ganju, Meher Kasam](https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html)
 - [Text similarity search with vector fields](https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch)
