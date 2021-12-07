from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json

import cv2
import os
import random

# from detectron2 import model_zoo
# import numpy as np
# import requests

def alternative_prepare_cfg(yaml_file, model_dir, model_name):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_dir
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)

    cfg.DATASETS.TRAIN = ("Planes_detection_Train",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset

    cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

    # cfg.INPUT.RANDOM_FLIP = "horizontal"
    # cfg.DATALOADER.NUM_WORKERS = 8
    # cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 100

    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[122.48, 158.97, 86.08, 71.7]]
    # # P3_C, KC_135, C_5, B_52
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
    #     [1.5, 2.17, 2.28, 1.83]
    #     ]
    return cfg


def prepare_config(yaml_file, model_dir, model_name):
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(yaml_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # Threshold

    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 48, 64, 128, 256]]
    # # P3_C, KC_135, C_5, B_52
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
    #     [1.0, 1.8, 2.17, 2.3]
    #     ]
    cfg.OUTPUT_DIR = model_dir
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.DEVICE = "cpu"  # cpu or cuda
    return cfg


def reg_dataset(name_of_dataset, imgs_root, f_path_annotation):
    # register_coco_instances(name_of_dataset, {}, f_path_annotation,imgs_root)
    load_coco_json(f_path_annotation, imgs_root, dataset_name=name_of_dataset)
    DatasetCatalog.register(name_of_dataset, lambda: load_coco_json(f_path_annotation, imgs_root, name_of_dataset))
    return name_of_dataset


def detecting_from_dir(testing_dir, saving_dir, name_of_dataset, cfg):
    print(MetadataCatalog.get(name_of_dataset))
    predictor = DefaultPredictor(cfg)
    imgs = []
    for i in os.listdir(testing_dir):
        if i[-3:] == "bmp":
            imgs.append(i)

    i = 0
    for img_name in imgs:
        image = cv2.imread(testing_dir + img_name)
        output = predictor(image)

        v = Visualizer(
            image[:, :, ::-1],
            metadata=MetadataCatalog.get(name_of_dataset),
            scale=0.8,
            instance_mode=ColorMode.SEGMENTATION,
        )
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        cv2.imwrite(saving_dir + img_name, v.get_image()[:, :, ::-1])
        print(f"Image {i} / {len(imgs)} is done.")
        i += 1
        # cv2.imshow("images", v.get_image()[:, :, ::-1])
        # cv2.waitKey(0)


def detecting_single_img(img_path, saving_dir, name_of_dataset, cfg):
    predictor = DefaultPredictor(cfg)
    img_name = img_path.split("/")[-1]
    image = cv2.imread(img_path)
    output = predictor(image)
    v = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get(name_of_dataset),
        scale=0.8,
        instance_mode=ColorMode.SEGMENTATION,
    )
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    cv2.imwrite(saving_dir + img_name, v.get_image()[:, :, ::-1])
    # cv2.imshow("images", v.get_image()[:, :, ::-1])
    # cv2.waitKey(0)


def main():
    yaml_file = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning_fixed_anchors/detectron2_config.yaml"
    model_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning_fixed_anchors"
    model_name = "model_final.pth"

    # name_of_dataset = "Planes_detection_Train"
    # imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_30k"
    # f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_30k/Train_Data.json"
    name_of_dataset = "Planes_detection_Train"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/Detectron2_dataset"
    f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/Detectron2_dataset/Train_Data_FULL.json"

    testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/temp_small/"
    saving_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/results_detectron_500epoches_CPU/"

    testing_img = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/Test_0.bmp"

    cfg = prepare_config(yaml_file, model_dir, model_name)
    # cfg = alternative_prepare_cfg(yaml_file, model_dir, model_name)
    # cfg = alternative_prepare_cfg("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning/detectron2_config.yaml", "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning", "model_0004999.pth")

    # cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0
    # cfg.MODEL.ASPECT_RATIOS = [[1.8, 2.17, 2.3]]
    # cfg.MODEL.SIZES = [[48, 64, 128, 256]]

    reg_dataset(name_of_dataset, imgs_root, f_path_annotation)

    # thing_colors = [
    #     tuple([random.randint(0, 255) for i in range(3)])
    #     for j in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    #     ]
    thing_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(thing_colors=thing_colors)
    stuff_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(stuff_colors=stuff_colors)

    # stuff_colors = [
    #     tuple([random.randint(0, 255) for i in range(3)])
    #     for j in range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    #     ]
    # MetadataCatalog.get(name_of_dataset).set(stuff_colors=stuff_colors)

    detecting_from_dir(testing_dir, saving_dir, name_of_dataset, cfg)
    # detecting_single_img(testing_img, saving_dir, name_of_dataset, cfg)


if __name__ == "__main__":
    main()
