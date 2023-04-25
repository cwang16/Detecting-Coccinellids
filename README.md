# Detecting-Coccinellids <br />
__Web-based application__ available at: https://coccinellids.cs.ksu.edu/ <br />
__Model__ used in the application: [model](https://ksuemailprod-my.sharepoint.com/:u:/g/personal/cwang16_ksu_edu/Ec2xffYhA9xLnBhGKEeDBAgBx-ENg0bkDil3i5Wdrw6eEQ?e=m0qs9w), all models in the paper:  [all models](https://ksuemailprod-my.sharepoint.com/:f:/g/personal/cwang16_ksu_edu/EkcBax9QZMdDnmnRWRfYZGIBH79HacgF4SeZd-frlog4oQ?e=ZdO2qs)<br />
__Annotations__ used in this paper: [annotations](https://ksuemailprod-my.sharepoint.com/:f:/g/personal/cwang16_ksu_edu/EgUTIZwyrNFJkJsBycLm-3kB3gHBfI0yfZLy964OEILu_g?e=fiBWc5) <br />
__Images__ used in this paper can be downloaded at: [iNaturalist Database](https://www.inaturalist.org/).  <br />
__Image IDs__ used in this paper can be downloaded at: [Image IDs](https://ksuemailprod-my.sharepoint.com/:f:/g/personal/cwang16_ksu_edu/EgWtcjs8XR5CiZDl9VEf3j4B5c5HgD7P4ZaRqAcXp1cyJQ?e=JmUyqP). 


##
![Web Image](https://i.pinimg.com/originals/9c/40/c6/9c40c67babece19e25859b736afe5fec.jpg) <br /> 
__DL-Coccinellids__ detects 7 classes (Coccinella_septempunctata, Coleomegilla_maculata, Cycloneda_sanguinea,Harmonia_axyridis, Hippodamia_convergens, Olla_nigrum, Scymninae) using the Faster R-CNN-FPN (Ren et al., 2015) implementation from Detectron2 (Wu et al., 2019). The Detectron2 framework is available at https://github.com/facebookresearch/detectron2. The Detectron2 code where the training object is  rewritten according to the Coccinellids objects detected is made available as part of this repository. We've also wrote visulizaiton code. Those modifications enable easy training or fine-tuning of models for other sets of Coccinellids images. In addition to Detectron2, we used the official implementations of YOLOv5 and YOLOv7, available at https://github.com/ultralytics/yolov5 and https://github.com/WongKinYiu/yolov7, respectively. 
<br />


## Content of the Repository
__train.py__ is the code for training the Faster R-CNN-FPN model for detecting Coccinellids in an image <br />
__test.py__ is the code for testing the Faster R-CNN-FPN model for detecting Coccinellids objects <br />
__MyTrainer.py__ cantains the customized training object <br />




## Prepare dataset:

1. prepare your dataset in COCO format

2. put the annotation in:

"./beetle/annotations/instances_train2017.json"

./beetle/annotations/instances_val2017.json"

3. put the images in:

./beetle/train2017

./beetle/val2017

## Install Detectron2:
Installation Instruction available at https://detectron2.readthedocs.io/en/latest/tutorials/install.html

## Merge files:
Put all the files in this repository in the root folder of detecton2

## Train:

1. specify the configuration file name in line 35 of train.py

2. specify the training and validation set path in lines 20 and 21. 

2. To save all validation result visualizations in the visualization folder, create an output/visualization folder in the main directory.

 

## Inference and visualization: 

1. To save all test result visualizations in the visualization folder, create an output/visualization folder in the main directory.

2. Put the model's name at line 57 of file test.py

3. Set the configuration file name in line 37 of test.py

## Yolov5:
Instructions are available at: https://github.com/ultralytics/yolov5


## Yolov7:
Instructions are available at: https://github.com/WongKinYiu/yolov7




