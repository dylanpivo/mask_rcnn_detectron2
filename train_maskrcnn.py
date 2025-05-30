from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

from config import get_detectron_cfg

register_coco_instances("my_train", {}, "annotations/train.json", "train")
register_coco_instances("my_val", {}, "annotations/val.json", "val")

_ = DatasetCatalog.get("my_train")
metadata = MetadataCatalog.get("my_train")
print("Classes:", metadata.thing_classes)

cfg = get_detectron_cfg(num_classes=len(metadata.thing_classes), train=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
