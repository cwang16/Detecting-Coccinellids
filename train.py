# check pytorch installation:
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from MyTrainer import MyTrainer
from detectron2.data.datasets import register_coco_instances
register_coco_instances("beetles_train", {}, "./beetle/annotations/instances_train2017.json", "./beetle/train2017")
register_coco_instances("beetles_val", {}, "./beetle/annotations/instances_val2017.json", "./beetle/val2017")

beetles_metadata = MetadataCatalog.get("beetles_train")
dataset_dicts = DatasetCatalog.get("beetles_train")


########################################################
###                  train_beetles                   ###
########################################################
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/your choice of architecture.yaml"))


cfg.DATASETS.TRAIN = ("beetles_train",)
cfg.DATASETS.TEST = ("beetles_val",)  # no metrics implemented for this dataset ##change
cfg.TEST.EVAL_PERIOD = 1000 # This will do evaluation once after 100 iterations on the cfg.DATASETS.TEST,
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0002
cfg.SOLVER.MAX_ITER = (
    10000
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # 7 classes (Coccinella_septempunctata, Coleomegilla_maculata, Cycloneda_sanguinea,
                                                 # Harmonia_axyridis, Hippodamia_convergens, Olla_nigrum, Scymninae)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)
from detectron2.utils.visualizer import ColorMode
cwd = os.getcwd()

model_image_vis_path = os.path.join(cwd, "output/visualization")
if not os.path.exists(model_image_vis_path):
    os.mkdir(os.path.join(cwd, "output/visualization"))
dataset_dicts = DatasetCatalog.get("beetles_val")
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=beetles_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
    )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    model_prediction = out.get_image()[:, :, ::-1]
    only_file_name= d["file_name"].split("/")[-1]
    saved_path = os.path.join(model_image_vis_path, only_file_name)
    cv2.imwrite(saved_path, model_prediction)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("beetles_val", ("bbox"), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "beetles_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
