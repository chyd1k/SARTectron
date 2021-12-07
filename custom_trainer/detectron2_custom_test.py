from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

import cv2
import os
import random
import torch
import numpy as np
from skimage import io

import json
import matplotlib

class CustomPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # fix
            # # Apply pre-processing to image.
            # if self.input_format == "RGB":
            #     # whether the model expects BGR inputs or RGB
            #     original_image = original_image[:, :, ::-1]

            # Convert one-dim to HWC
            # duplicate grayscale to 3 channels
            # if len(original_image.shape) == 2:
            #     original_image = np.stack((original_image,) * 3, axis=-1)

            height, width = original_image.shape[:2]
            original_image = original_image.astype("float32")
            image = self.aug.get_transform(original_image).apply_image(original_image)
            # image = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            image = torch.as_tensor(image.astype("float32"))

            print(image.shape)
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def alternative_prepare_cfg(yaml_file, model_dir, model_name):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = model_dir
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)

    cfg.DATASETS.TRAIN = ("coco_Planes_detection_Train",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset

    cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000  # 5000 originally 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000  # 5000 originally 1000
    cfg.TEST.DETECTIONS_PER_IMAGE = 200

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    return cfg


def prepare_config(yaml_file, model_dir, model_name):
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(yaml_file)

    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 5000  # 5000 originally 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000  # 5000 originally 1000
    cfg.TEST.DETECTIONS_PER_IMAGE = 200

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # Threshold
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    # cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.5]

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
    # predictor = DefaultPredictor(cfg)
    predictor = CustomPredictor(cfg)
    imgs = []
    for i in os.listdir(testing_dir):
        if (i[-3:] == "bmp") or (i[-3:] == "tif"):
            imgs.append(i)

    i = 0
    for img_name in imgs:
        image = utils.read_image(testing_dir + img_name)
        # if len(image.shape) == 2:
        #     image = np.stack((image,) * 3, axis=-1)

        # image = utils.read_image('C:/Users/savchenko.bs/Desktop/' + 'HG9RaUb7kH4.jpg')
        # image = image.astype("float32")
        output = predictor(image)

        # image = cv2.imread(testing_dir + img_name)
        # output = predictor(image)
        if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)

        v = Visualizer(
            image[:, :, ::-1],
            metadata=MetadataCatalog.get(name_of_dataset),
            # scale=0.8,
            instance_mode=ColorMode.SEGMENTATION,
        )
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        cv2.imshow("images", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.imwrite(saving_dir + img_name, v.get_image()[:, :, ::-1])
        print(f"Image {i} / {len(imgs)} is done.")
        i += 1


def detecting_single_img(img_path, saving_dir, name_of_dataset, cfg):
    predictor = CustomPredictor(cfg)
    img_name = img_path.split("/")[-1]
    image = utils.read_image(img_path, format="RGB")
    # image = image.astype("float32")
    output = predictor(image)
    print(image.dtype)
    v = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get(name_of_dataset),
        # scale=0.8,
        instance_mode=ColorMode.SEGMENTATION,
    )
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    # cv2.imwrite(saving_dir + img_name, v.get_image().astype("float32"))
    cv2.imshow("images", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


def plot_metrics(experiment_folder):
    matplotlib.use( 'tkagg' )
    json_path = experiment_folder + '/metrics.json'
    experiment_metrics = []
    with open(json_path, 'r') as f:
        for line in f:
            experiment_metrics.append(json.loads(line))

    matplotlib.pyplot.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
    matplotlib.pyplot.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    matplotlib.pyplot.legend(['total_loss', 'validation_loss'], loc='upper left')
    matplotlib.pyplot.show()


    fig, ax1 = matplotlib.pyplot.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')

    ax1.plot(
        [x['iteration'] for x in experiment_metrics  if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color="black", label="Total Loss")
    ax1.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], color="dimgray", label="Val Loss")

    ax1.tick_params(axis='y')
    matplotlib.pyplot.legend(loc='upper left')

    ax2 = ax1.twinx()

    color = 'tab:orange'
    ax2.set_ylabel('AP')
    ax2.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['bbox/AP'] for x in experiment_metrics if 'bbox/AP' in x], color=color, label="AP")
    ax2.tick_params(axis='y')

    matplotlib.pyplot.legend(loc='upper right')
    matplotlib.pyplot.show()


def visualize_img(name_of_dataset, ind_img):
    res = DatasetCatalog.get(name_of_dataset)
    if ind_img > len(res):
        return
    if res[ind_img]["file_name"].lower().endswith(('.tiff', '.tif')):
        img = io.imread(res[ind_img]["file_name"])
    else:
        img = cv2.imread(res[ind_img]["file_name"])
    print(img.shape)
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    visualizer = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get(name_of_dataset),
        scale=0.8
        )
    vis = visualizer.draw_dataset_dict(res[ind_img])
    cv2.imshow(f"Image #{ind_img}", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    return


def test_best():
    yaml_file = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_09_04/detectron2_config.yaml"
    model_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_09_04"
    model_name = "model_0010749.pth"
    cfg = prepare_config(yaml_file, model_dir, model_name)

    name_of_dataset = "coco_Planes_detection_Train"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k"
    f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data.json"
    reg_dataset(name_of_dataset, imgs_root, f_path_annotation)

    thing_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(thing_colors=thing_colors)
    stuff_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(stuff_colors=stuff_colors)

    testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/"
    saving_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_09_04/results_detection/"

    saving_dir = "C:/Users/savchenko.bs/Desktop/new_test_Detectron2/TIFF_TEST/results_detection/"
    testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/"
    # saving_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_09_04/results_detection/13499/"
    saving_dir = "C:/Users/savchenko.bs/Desktop/Demonstration/Results_on_testing_dataset/Detectron2_10749_iter/Treshhold_09/"
    # testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/Test_TIFF/8000_12000_GRAYSCALE_TIFF"

    # detecting_from_dir("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/Test_TIFF/", saving_dir, name_of_dataset, cfg)
    detecting_from_dir(testing_dir + "/", saving_dir, name_of_dataset, cfg)
    return


def test_08_06():
    yaml_file = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_08_06/detectron2_config.yaml"
    model_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_08_06"
    model_name = "model_0005749.pth"
    cfg = prepare_config(yaml_file, model_dir, model_name)

    name_of_dataset = "coco_Planes_detection_Train"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k"
    f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data.json"
    reg_dataset(name_of_dataset, imgs_root, f_path_annotation)

    thing_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(thing_colors=thing_colors)
    stuff_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(stuff_colors=stuff_colors)

    testing_dir = "C:/Users/savchenko.bs/Desktop/new_test_Detectron2/TIFF_TEST/"
    saving_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_08_06/results_detection/"
    testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/"

    # detecting_from_dir("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/Test_TIFF/", saving_dir, name_of_dataset, cfg)
    detecting_from_dir(testing_dir + "/", saving_dir, name_of_dataset, cfg)
    return


def main():
    test_best()
    # test_08_06()

    # # Рабочая комбинация
    # yaml_file = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning_new/detectron2_config.yaml"
    # model_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning"
    # model_name = "model_0004999.pth"

    yaml_file = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_09_04/detectron2_config.yaml"
    model_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_09_04"
    model_name = "model_0013249.pth"

    # name_of_dataset = "Planes_detection_Train"
    # imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_30k"
    # f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_30k/Train_Data.json"
    name_of_dataset = "coco_Planes_detection_Train"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k"
    f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data.json"

    test_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json"

    # testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/temp_small/"
    # testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/temp_big/"
    testing_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/"

    # saving_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/custom_trainer/res_26_03/"
    saving_dir = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_09_04/results_detection/"

    # testing_img = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/custom_trainer/test.tif"
    testing_img = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/0.tif"

    # plot_metrics("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_24_03")


    cfg = prepare_config(yaml_file, model_dir, model_name)
    # cfg = alternative_prepare_cfg(yaml_file, model_dir, model_name)
    # cfg = alternative_prepare_cfg("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning/detectron2_config.yaml", "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning", "model_0004999.pth")


    reg_dataset(name_of_dataset, imgs_root, f_path_annotation)
    # name_of_test_dataset = "coco_Planes_detection_Test"
    # imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF"
    # test_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json"
    # reg_dataset(name_of_test_dataset, imgs_root, test_path_annotation)
    # visualize_img(name_of_test_dataset, 5)
    # visualize_img(name_of_test_dataset, 6)

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

    # detecting_from_dir("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/Test_TIFF/", saving_dir, name_of_dataset, cfg)
    # detecting_from_dir("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/test_dataset_tiff/", saving_dir, name_of_dataset, cfg)


    # detecting_from_dir("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/Test_TIFF/8000_12000_GRAYSCALE_TIFF/", saving_dir, name_of_dataset, cfg)
    detecting_from_dir("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/Test/Test_TIFF/", saving_dir, name_of_dataset, cfg)
    detecting_from_dir(testing_dir + "/", saving_dir, name_of_dataset, cfg)


    # detecting_single_img(testing_img, saving_dir, name_of_dataset, cfg)
    # detecting_single_img("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_0.tif", saving_dir, name_of_dataset, cfg)
    # detecting_single_img("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_1.tif", saving_dir, name_of_dataset, cfg)
    # detecting_single_img("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_2.tif", saving_dir, name_of_dataset, cfg)
    # detecting_single_img("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_3.tif", saving_dir, name_of_dataset, cfg)
    # detecting_single_img("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_4.tif", saving_dir, name_of_dataset, cfg)

    # detecting_single_img("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_5.tif", saving_dir, name_of_dataset, cfg)

    # detecting_single_img("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_1843.tif", saving_dir, name_of_dataset, cfg)


if __name__ == "__main__":
    main()
