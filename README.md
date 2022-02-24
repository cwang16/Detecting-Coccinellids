# Detecting-Coccinellids-Using-Faster-RCNN
Web-based application available at: https://coccinellids.cs.ksu.edu/ <br />
##
__DL-Coccinellids__ detects 7 classes (Coccinella_septempunctata, Coleomegilla_maculata, Cycloneda_sanguinea,
                                                 # Harmonia_axyridis, Hippodamia_convergens, Olla_nigrum, Scymninae) using Detectron2 (Wu et al., 2019)  - Faster RCNN (Ren et al., 2015). The Detectron2 is available at https://github.com/facebookresearch/detectron2. The Python/TensforFlow implementation of Faster R-CNN (Chen and Gupta, 2017), available at https://github.com/endernewton/tf-faster-rcnn, was used to build two models for root anatomy. The first model detects root and stele objects, and the second one detects late metaxylem objects. The models are trained with around 300 images, including our own dataset and several images from RootAnalyzer (10 images) and RootCell (6 images).  The code where the last layer is modified according to the root objects detected is made available here to enable easy training or fine-tuning of models for other sets of root cross-section images. 
<br />
![Web Image](https://i.pinimg.com/originals/9c/40/c6/9c40c67babece19e25859b736afe5fec.jpg) <br /> 

## Content of the Repository
__train.py__ is the code for training the model for detection the Coccinellids in an image <br />
__test.py__ is the code for testing the model for detecting theCoccinellids objects <br />
__model__ folder cantains the model trained with about 300 images, <br />


## Prepare dataset:

1. prepare your dataset in COCO format

2. put the annotation in:

"./beetle/annotations/instances_train2017.json"

./beetle/annotations/instances_val2017.json"

3. put the images in:

./beetle/train2017

./beetle/val2017



## Train:

1. specify the configuration file name in line 35 of train.py

2. specify the training and validation set path in line 20 and 21. 

2. To save all validation result visualizations in the visualization folder, create an output/visualization folder in the main directory.

 

## Inference and visualization: 

1. To save all test result visualizations in the visualization folder, create an output/visualization folder in the main directory.

2. Put the model's name at line 57 of file test.py

3. Set the configuration file name in line 37 of test.py



