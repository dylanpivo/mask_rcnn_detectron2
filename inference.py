import cv2
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

from config import get_detectron_cfg

register_coco_instances("my_train", {}, "annotations/train.json", "train")

cfg = get_detectron_cfg(num_classes=2, test_threshold=0.2, train=False)

predictor = DefaultPredictor(cfg)

_ = DatasetCatalog.get("my_train")
metadata = MetadataCatalog.get("my_train")
print("Classes:", metadata.thing_classes)

image_path = "./val/th (10).jpg"
im = cv2.imread(image_path)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=MetadataCatalog.get("my_train"),
               scale=0.5,
               instance_mode=ColorMode.IMAGE_BW
               )

out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
cv2.waitKey(0)

instances = outputs["instances"]
boxes = instances.pred_boxes if instances.has("pred_boxes") else None
masks = instances.pred_masks if instances.has("pred_masks") else None
classes = instances.pred_classes if instances.has("pred_classes") else None
scores = instances.scores if instances.has("scores") else None
