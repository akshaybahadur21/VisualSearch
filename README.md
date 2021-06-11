# VisualSearch 🖼️ 🔍

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Emojinator/blob/master/LICENSE.md)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

A Reverse Image Search Engine pipeline for humans

## Code Requirements 🦄
You can install Conda for python which resolves all the dependencies for machine learning.

`pip install -r requirements.txt`

## Description 🌈

Reverse image search (or as it is more technically known, instance retrieval) enables developers and researchers to build scenarios beyond simple keyword search. From discovering visually similar objects on Pinterest to recommending similar songs on Spotify to camera-based product search on Amazon, a similar class of technology under the hood is used. Sites like TinEye alert photographers on copyright infringement when their photographs are posted without consent on the internet. Even face recognition in several security systems uses a similar concept to ascertain the identity of the person.

## Modules
- Image Similarity
- Feature Extraction
- Similarity Search


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
