import os
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo


def get_detectron_cfg(num_classes=1, test_threshold=0.7, train=False):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))

    if train:
        cfg.DATASETS.TRAIN = ("my_train",) if train else ()
        cfg.DATASETS.TEST = ("my_val",) if train else ("my_train",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ) if train else os.path.join("./output", "model_final.pth")

    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg
