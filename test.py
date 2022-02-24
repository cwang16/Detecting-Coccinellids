# check pytorch installation:
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from MyTrainer import MyTrainer

##################################
###            test            ###
##################################
from detectron2.data.datasets import register_coco_instances
register_coco_instances("beetles_train", {}, "./beetle/annotations/instances_train2017.json", "./beetle/train2017")
register_coco_instances("beetles_val", {}, "./beetle/annotations/instances_test2017.json", "./beetle/test2017")
beetles_metadata = MetadataCatalog.get("beetles_train")
dataset_dicts = DatasetCatalog.get("beetles_train")


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/your choice of architecture.yaml"))


cfg.DATASETS.TRAIN = ("beetles_train",)
cfg.DATASETS.TEST = ("beetles_val",)
cfg.TEST.EVAL_PERIOD = 5000
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0002
cfg.SOLVER.MAX_ITER = (
    400000
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # 7 classes (Coccinella_septempunctata, Coleomegilla_maculata, Cycloneda_sanguinea,
                                                 # Harmonia_axyridis, Hippodamia_convergens, Olla_nigrum, Scymninae)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg) # use MyTrainer to add validation graph

model_name = "model_name.pth"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
cwd = os.getcwd()

model_image_vis_path = os.path.join(cwd, "output/visualization", model_name)

if not os.path.exists(model_image_vis_path):
    os.mkdir(os.path.join(cwd, "output/visualization", model_name))
dataset_dicts = DatasetCatalog.get("beetles_val")
cocci = [0,0,0,0,0,0,0,0]
cole = [0,0,0,0,0,0,0,0]
cycl = [0,0,0,0,0,0,0,0]
harm = [0,0,0,0,0,0,0,0]
hippo = [0,0,0,0,0,0,0,0]
olla = [0,0,0,0,0,0,0,0]
scym = [0,0,0,0,0,0,0,0]
nothing = [0,0,0,0,0,0,0,0]
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    ground_truth_count = len(d["annotations"])
    outputs = predictor(im)
    score = outputs['instances']
    lens_prediction = len(outputs['instances'])# format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print (d['file_name'])
    print (lens_prediction, ground_truth_count)
    v = Visualizer(im[:, :, ::-1],
                   metadata=beetles_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    model_prediction = out.get_image()[:, :, ::-1]
    only_file_name= d["file_name"].split("/")[-1]
    saved_path = os.path.join(model_image_vis_path, only_file_name)
    cv2.imwrite(saved_path, model_prediction)

#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("beetles_val", cfg, True, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "beetles_val")

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)

