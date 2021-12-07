import os, sys, torch, cv2, copy
import numpy as np
from skimage import io
from LossEvalHook import LossEvalHook
from optparse import OptionParser
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# import json
# from detectron2.structures import BoxMode
# import detectron2.data.transforms as T


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


def custom_mapper(dataset_list):
    dataset_list = copy.deepcopy(dataset_list)  # it will be modified by code below
    l = len(dataset_list)

    image = utils.read_image(dataset_list["file_name"], format=None)
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
    # dataset_list["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    dataset_list["image"] = torch.as_tensor(image.astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, [], image.shape[:2])
        for obj in dataset_list.pop("annotations")
        if obj.get("iscrowd", 0) == 0
        ]
    dataset_list["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
    # instances = utils.annotations_to_instances(annos, image.shape[:2])
    # dataset_list["instances"] = utils.filter_empty_instances(instances)
    return dataset_list


def create_cfg(weights_root, name_of_dataset_train, name_of_dataset_test):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = weights_root

    cfg.DATASETS.TRAIN = (name_of_dataset_train,)
    if name_of_dataset_test == "":
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = (name_of_dataset_test,)
        cfg.TEST.EVAL_PERIOD = 25 # 50

    cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

    # cfg.INPUT.RANDOM_FLIP = "horizontal"
    # cfg.DATALOADER.NUM_WORKERS = 8
    # cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.MODEL.PIXEL_MEAN = [0.0] # len(PIXEL_MEAN) -> input_shape C:\Users\savchenko.bs\Desktop\new_placement\detectron2\detectron2\layers\shape_spec.py
    cfg.MODEL.PIXEL_STD = [1.0]

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    # Total number of RoIs per training minibatch = ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    # E.g., a common configuration is: 512 * 16 = 8192
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # 512
    cfg.SOLVER.IMS_PER_BATCH = 5 # 16

    cfg.SOLVER.CHECKPOINT_PERIOD = 250

    cfg.SOLVER.BASE_LR = 0.01 # 0.003 # 0.0025
    cfg.SOLVER.MAX_ITER = 15000 # 2000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000

    cfg.TEST.DETECTIONS_PER_IMAGE = 200

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


def reg_dataset(name, imgs_folder, annotation_path):
    dcs = load_coco_json(annotation_path, imgs_folder, dataset_name=name)
    DatasetCatalog.register(name, lambda: load_coco_json(annotation_path, imgs_folder, name))
    return dcs


def data_preprocessing(
        name_train_dataset, name_test_dataset,
        train_imgs_folder, test_imgs_folder,
        train_annotation_path, test_annotation_path
        ):
    train_dcs = reg_dataset(name_train_dataset, train_imgs_folder, train_annotation_path)
    test_dcs = reg_dataset(name_test_dataset, test_imgs_folder, test_annotation_path)
    return train_dcs, test_dcs


def config_preprocessing(outp_weights_path, name_train_dataset, name_test_dataset):
    # config params function
    cfg = create_cfg(outp_weights_path, name_train_dataset, name_test_dataset)
    # write config
    cfg_name = "detectron2_config.yaml"
    cfg_name = write_cfg(cfg, outp_weights_path + "/" + cfg_name)
    return cfg


def get_cfg_from_file(path_yaml):
    cfg = get_cfg()
    cfg.merge_from_file(path_yaml)
    return cfg


def try_preparation():
    # registration dataset
    train_dcs, test_dcs = data_preprocessing(
        "coco_Planes_detection_Train",
        "coco_Planes_detection_Test",
        "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k",
        "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k",
        "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data_with_no_objects.json",
        "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json"
        )

    cfg = config_preprocessing(
        "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_08_06",
        "coco_Planes_detection_Train", "coco_Planes_detection_Test"
        )
    return train_dcs, test_dcs, cfg


def useful_funcs():
    # open cfg from file
    name_train_dataset = "coco_Planes_detection_Train"
    name_test_dataset = "coco_Planes_detection_Test"
    train_imgs_folder = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k"
    test_imgs_folder = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF"
    train_annotation_path = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data_with_no_objects.json"
    test_annotation_path = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json"
    outp_weights_path = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_08_06"
    # registration dataset
    train_dcs, test_dcs = data_preprocessing(
        name_train_dataset, name_test_dataset,
        train_imgs_folder, test_imgs_folder,
        train_annotation_path, test_annotation_path
        )
    cfg = config_preprocessing(outp_weights_path, name_train_dataset, name_test_dataset)

    cfg_name = "detectron2_config.yaml"
    cfg = get_cfg_from_file("C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_08_06" + "/" + cfg_name)
    visualize_img(name_test_dataset, 5)
    planes_metadata = MetadataCatalog.get(name_train_dataset)
    print(planes_metadata)
    custom_mapper(train_dcs)
    # write weights from specific config
    name_model = write_weights_from_cfg(cfg, outp_weights_path,"detectron2_model")


def train_begin(
        name_train_dataset, name_test_dataset,
        train_imgs_folder, test_imgs_folder,
        train_annotation_path, test_annotation_path,
        outp_weights_path
        ):
    # train_dcs, test_dcs, cfg = try_preparation()

    # registration dataset
    train_dcs, test_dcs = data_preprocessing(
        name_train_dataset, name_test_dataset,
        train_imgs_folder, test_imgs_folder,
        train_annotation_path, test_annotation_path
        )

    cfg = config_preprocessing(outp_weights_path, name_train_dataset, name_test_dataset)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


def parse_params():
    sys.setrecursionlimit(40000)
    parser = OptionParser()

    parser.add_option("-n_tr", "--name_train_ds", dest="name_train_ds", help="Name of training dataset.")
    parser.add_option("-n_tst", "--name_test_ds", dest="name_test_ds", help="Name of testing dataset.")
    parser.add_option("-tr_f", "--train_folder", dest="train_folder", help="Path to folder with training data.")
    parser.add_option("-tst_f", "--test_folder", dest="test_folder", help="Path to folder with testing data.")
    parser.add_option("-tr_ann_p", "--train_ann_path", dest="train_ann_path", help="Path to train annotation json-file.")
    parser.add_option("-tst_ann_p", "--test_ann_path", dest="test_ann_path", help="Path to test annotation json-file.")
    parser.add_option("-saving_f", "--saving_folder", dest="saving_folder", help="Path to folder where weights will be saved.")
    (options, args) = parser.parse_args()

    name_train_dataset = options.name_train_ds
    name_test_dataset = options.name_test_ds
    train_imgs_folder = options.train_folder
    test_imgs_folder = options.test_folder
    train_annotation_path = options.train_ann_path
    test_annotation_path = options.test_ann_path
    outp_weights_path = options.saving_folder
    
    train_begin(
        name_train_dataset, name_test_dataset,
        train_imgs_folder, test_imgs_folder,
        train_annotation_path, test_annotation_path,
        outp_weights_path
        )


def main():
    # train_dcs, test_dcs = data_preprocessing(
    #     "coco_Planes_detection_Train",
    #     "coco_Planes_detection_Test",
    #     "C:/Users/savchenko.bs/Desktop/AnnotationToolTest",
    #     "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF",
    #     "C:/Users/savchenko.bs/Desktop/AnnotationToolTest/Train_Data.json",
    #     "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json"
    #     )

    # for i in range(300, 500):
    #     visualize_img("coco_Planes_detection_Train", i)
    #     cv2.waitKey(0)
    
    # parse_params()
    # train_begin("coco_Planes_detection_Train",
    #     "coco_Planes_detection_Test",
    #     "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k",
    #     "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF",
    #     "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data_with_no_objects.json",
    #     "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json",
    #     "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/res_learning_14_07"
    # )
    train_begin("coco_Planes_detection_Train",
        "coco_Planes_detection_Test",
        "C:/Users/savchenko.bs/Desktop/AnnotationToolTest",
        "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF",
        "C:/Users/savchenko.bs/Desktop/AnnotationToolTest/Train_Data.json",
        "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json",
        "C:/Users/savchenko.bs/Desktop/AnnotationToolTest/weights"
    )


if __name__ == "__main__":
    main()
