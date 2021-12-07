import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.config import get_cfg
import copy
import torch
import cv2

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def custom_mapper(dataset_list):
    dataset_list = copy.deepcopy(dataset_list)  # it will be modified by code below
    l = len(dataset_list)

    image = utils.read_image(dataset_list["file_name"], format="BGR")
    # transform_list = [
    #     T.Resize((800,800))
    #     # T.Resize((800,800)),
    #     # T.RandomBrightness(0.8, 1.8),
    #     # T.RandomContrast(0.6, 1.3),
    #     # T.RandomSaturation(0.8, 1.4),
    #     # T.RandomRotation(angle=[90, 90]),
    #     # T.RandomLighting(0.7),
    #     # T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    #     ]
    # image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_list["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, [], image.shape[:2])
        for obj in dataset_list.pop("annotations")
        if obj.get("iscrowd", 0) == 0
        ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_list["instances"] = utils.filter_empty_instances(instances)
    return dataset_list


def create_cfg(weights_root, name_of_dataset):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = weights_root

    cfg.DATASETS.TRAIN = (name_of_dataset,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset

    cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

    # cfg.INPUT.RANDOM_FLIP = "horizontal"
    # cfg.DATALOADER.NUM_WORKERS = 8
    # cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.CHECKPOINT_PERIOD = 500

    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[122.48, 158.97, 86.08, 71.7]]
    # # P3_C, KC_135, C_5, B_52
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
    #     [1.5, 2.17, 2.28, 1.83]
    #     ]

    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
    #     [0.5, 1.0, 1.8, 2.17, 2.3]
    #     ]

    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
    #     [1.0, 1.8], [1.0, 2.3], [1.0, 2.17]
    #     ]
    return cfg


def write_cfg(cfg, full_cfg_path):
    with open(full_cfg_path, "w") as f:
        f.write(cfg.dump())
    return full_cfg_path


def write_weights_from_cfg(cfg, saving_dir, weights_name):
    cfg.OUTPUT_DIR = saving_dir
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir=saving_dir)
    checkpointer.save(weights_name)
    return saving_dir + "/" + weights_name


def visualize_img(name_of_dataset, ind_img):
    res = DatasetCatalog.get(name_of_dataset)
    if ind_img > len(res):
        return
    img = cv2.imread(res[ind_img]["file_name"])
    visualizer = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get(name_of_dataset),
        scale=0.5
        )
    vis = visualizer.draw_dataset_dict(res[ind_img])
    cv2.imshow(f"Image #{ind_img}", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    return


def main():
    # registration dataset
    name_of_dataset = "coco_Planes_detection_Train"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k"
    f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data.json"

    list_of_dicts = load_coco_json(f_path_annotation, imgs_root, dataset_name=name_of_dataset)
    DatasetCatalog.register(name_of_dataset, lambda: load_coco_json(f_path_annotation, imgs_root, name_of_dataset))
    planes_metadata = MetadataCatalog.get(name_of_dataset)
    print(planes_metadata)
    # custom_mapper(list_of_dicts)
    ######################

    # config params function
    weights_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_24_03"
    cfg = create_cfg(weights_root, name_of_dataset)
    # write weights from specific config
    # name_model = write_weights_from_cfg(cfg, weights_root,"detectron2_model")
    ########################

    # write config
    cfg_name = "detectron2_config.yaml"
    cfg_name = write_cfg(cfg, weights_root + "/" + cfg_name)
    ##############

    # open cfg from file
    cfg_from_file = get_cfg()
    cfg_from_file.merge_from_file(cfg_name)
    ####################

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
