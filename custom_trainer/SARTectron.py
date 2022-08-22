def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
from turtle import shape
import torch.jit
torch.jit.script_method = script_method
torch.jit.script = script

import os, sys, torch, gc, time, cv2, copy, re, json
# sys.path.append('D:/detectron2/detectron2')

import numpy as np
from LossEvalHook import LossEvalHook
from optparse import OptionParser

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.catalog import Metadata
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T


gc.collect()
torch.cuda.empty_cache()


def custom_mapper(dataset_list):
    dataset_list = copy.deepcopy(dataset_list)  # it will be modified by code below

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
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_list["instances"] = utils.filter_empty_instances(instances)
    return dataset_list


class CustomTrainerAndVal(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

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


class CustomTrainerNoVal(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


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

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def compute_iou(box, box2, min_w, min_h):
    xA = max(box[0], box2[0])
    yA = max(box[1], box2[1])
    xB = min(box[2], box2[2])
    yB = min(box[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)        
    box1Area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = 1 if (box1Area < 10 or box2Area < 10 or
        box[2] - box[0] < min_w or box[3] - box[1] < min_h or
        box2[2] - box2[0] < min_w or box2[3] - box2[1] < min_h) else interArea / float(box1Area + box2Area - interArea)
    return iou


# { "class" : [ {"bbox", "prob", "path"}, ... ],
#   "class2" :  [ {"bbox", "prob", "path"}    ] }
def non_maximum_suppression(js, threshhold, min_w, min_h):
    temp_dict = {}
    for i in js:
        for j in js[i]:
            if j["class_name"] not in temp_dict.keys():
                temp_dict[j["class_name"]] = []
            j["path"] = i
            class_name = j["class_name"]
            del j["class_name"]
            temp_dict[class_name].append(j)

    final_dict = {}
    for i in temp_dict:
        final_boxes = []
        lst_json_sorted = sorted(temp_dict[i], key=lambda d: d["prob"], reverse=True)
        while len(lst_json_sorted) > 0:
            # removing the best probability bounding box
            box = lst_json_sorted.pop(0)
            for b in lst_json_sorted:
                iou = compute_iou(box["bbox"], b["bbox"], min_w, min_h)
                if iou >= threshhold:
                    lst_json_sorted.remove(b)
            final_boxes.append(box)
        for j in final_boxes:
            if j["path"] not in final_dict:
                final_dict[j["path"]] = []
            final_dict[j["path"]].append({"bbox" : j["bbox"], "class_name" : i, "prob" : j["prob"]})
    return final_dict


def write_cfg(cfg, full_cfg_path):
    with open(full_cfg_path, "w") as f:
        f.write(cfg.dump())
    return full_cfg_path


def reg_dataset(name, imgs_folder, annotation_path):
    dcs = load_coco_json(annotation_path, imgs_folder, dataset_name=name)    
    thing_colors = [tuple(np.random.choice(range(256), size=3)) for _ in MetadataCatalog.get(name).thing_classes]
    metadata = {
        "thing_colors" : thing_colors
    }
    register_coco_instances(name, metadata, annotation_path, imgs_folder)
    return dcs


def set_cfg_params(params, base_cfg_path = ""):
    cfg = get_cfg()

    if base_cfg_path != "":
        cfg.merge_from_file(base_cfg_path)

    cfg.MODEL.WEIGHTS = params["MODEL_WEIGHTS"]
    cfg.OUTPUT_DIR = params["OUTPUT_DIR"]
    cfg.DATASETS.TRAIN = (params["NAME_OF_TRAIN_DATASET"],)
    if params["NAME_OF_TEST_DATASET"] == "":
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = (params["NAME_OF_TEST_DATASET"],)
        cfg.TEST.EVAL_PERIOD = params["TEST_EVAL_PERIOD"]
    cfg.MODEL.DEVICE = params["MODEL_DEVICE"]
    cfg.MODEL.PIXEL_MEAN = [0.0]
    cfg.MODEL.PIXEL_STD = [1.0]
    cfg.INPUT.MIN_SIZE_TRAIN = params["INPUT_MIN_SIZE_TRAIN"] 
    cfg.INPUT.MAX_SIZE_TRAIN = params["INPUT_MAX_SIZE_TRAIN"] 
    cfg.INPUT.MIN_SIZE_TEST = params["INPUT_MIN_SIZE_TEST"] 
    cfg.INPUT.MAX_SIZE_TEST = params["INPUT_MAX_SIZE_TEST"] 
    cfg.DATALOADER.NUM_WORKERS = params["DATALOADER_NUM_WORKERS"]
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = params["DATALOADER_FILTER_EMPTY_ANNOTATIONS"]
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = params["MODEL_RPN_BATCH_SIZE_PER_IMAGE"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params["MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE"]
    cfg.MODEL.RPN.POSITIVE_FRACTION = params["MODEL_RPN_POS_FRACTION"]
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = params["MODEL_ROI_HEADS_POS_FRACTION"]
    cfg.SOLVER.IMS_PER_BATCH = params["SOLVER_IMS_PER_BATCH"]
    cfg.SOLVER.CHECKPOINT_PERIOD = params["SOLVER_CHECKPOINT_PERIOD"]
    cfg.SOLVER.BASE_LR = params["SOLVER_BASE_LR"]
    cfg.SOLVER.MAX_ITER = params["SOLVER_MAX_ITER"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params["MODEL_ROI_HEADS_NUM_CLASSES"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = params["MODEL_RPN_PRE_NMS_TOPK_TRAIN"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = params["MODEL_RPN_PRE_NMS_TOPK_TEST"]
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = params["MODEL_RPN_POST_NMS_TOPK_TRAIN"]
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = params["MODEL_RPN_POST_NMS_TOPK_TEST"]
    cfg.TEST.DETECTIONS_PER_IMAGE = params["TEST_DETECTIONS_PER_IMAGE"]
    cfg.MODEL.ANCHOR_GENERATOR.EXPECTED_SHAPES = params["ANCHOR_GENERATOR_EXPECTED_SHAPES"]
    cfg.DATASETS.CLASSES_NAMES = params["CLASSES_NAMES"]
    cfg.MODEL.RADAR_NMS = params["RADAR_NMS"]
    # MetadataCatalog.get(params["NAME_OF_TRAIN_DATASET"]).thing_classes
    return cfg


def TrainBegin(options):
    train_imgs_folder = options.train_folder
    test_imgs_folder = options.test_folder
    train_annotation_path = options.train_ann_path
    test_annotation_path = options.test_ann_path

    name_train_dataset = options.NAME_OF_TRAIN_DATASET
    name_test_dataset = options.NAME_OF_TEST_DATASET
    outp_weights_path = options.OUTPUT_DIR
    base_cfg_path = options.BASE_CFG_PATH

    # options.ANCHOR_GENERATOR_EXPECTED_SHAPES = "[[100, 150], [108, 234], [57, 130], [53, 97]]"
    if (len(options.ANCHOR_GENERATOR_EXPECTED_SHAPES) == 0):
        print("Pass expected object shapes to program with --ANCHOR_GENERATOR_EXPECTED_SHAPES command.", flush=True)
        return
    
    if (len(options.CLASSES_NAMES) == 0) or (len(options.CLASSES_NAMES) != int(options.MODEL_ROI_HEADS_NUM_CLASSES)):
        print("Pass classes names and number of classes via --CLASSES_NAMES and --MODEL_ROI_HEADS_NUM_CLASSES commands.", flush=True)
        return

    params = {
        "MODEL_WEIGHTS" : options.MODEL_WEIGHTS,

        "NAME_OF_TRAIN_DATASET" : options.NAME_OF_TRAIN_DATASET,
        "NAME_OF_TEST_DATASET" : options.NAME_OF_TEST_DATASET,
        "OUTPUT_DIR" : options.OUTPUT_DIR,
        "BASE_CFG_PATH" : options.BASE_CFG_PATH,

        "TEST_EVAL_PERIOD" : options.TEST_EVAL_PERIOD,
        "MODEL_DEVICE" : options.MODEL_DEVICE,
        "INPUT_MIN_SIZE_TRAIN" : (options.INPUT_MIN_SIZE_TRAIN,),
        "INPUT_MAX_SIZE_TRAIN" : options.INPUT_MAX_SIZE_TRAIN,
        "INPUT_MIN_SIZE_TEST" : options.INPUT_MIN_SIZE_TEST,
        "INPUT_MAX_SIZE_TEST" : options.INPUT_MAX_SIZE_TEST,
        "DATALOADER_NUM_WORKERS" : options.DATALOADER_NUM_WORKERS,
        "DATALOADER_FILTER_EMPTY_ANNOTATIONS" : options.DATALOADER_FILTER_EMPTY_ANNOTATIONS,
        "MODEL_RPN_BATCH_SIZE_PER_IMAGE" : options.MODEL_RPN_BATCH_SIZE_PER_IMAGE,
        "MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE" : options.MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE,
        "MODEL_RPN_POS_FRACTION" : options.MODEL_RPN_POS_FRACTION,
        "MODEL_ROI_HEADS_POS_FRACTION" : options.MODEL_ROI_HEADS_POS_FRACTION,
        "SOLVER_IMS_PER_BATCH" : options.SOLVER_IMS_PER_BATCH,
        "SOLVER_CHECKPOINT_PERIOD" : options.SOLVER_CHECKPOINT_PERIOD,
        "SOLVER_BASE_LR" : options.SOLVER_BASE_LR,
        "SOLVER_MAX_ITER" : options.SOLVER_MAX_ITER,
        "MODEL_ROI_HEADS_NUM_CLASSES" : options.MODEL_ROI_HEADS_NUM_CLASSES,
        "MODEL_RPN_PRE_NMS_TOPK_TRAIN" : options.MODEL_RPN_PRE_NMS_TOPK_TRAIN,
        "MODEL_RPN_PRE_NMS_TOPK_TEST" : options.MODEL_RPN_PRE_NMS_TOPK_TEST,
        "MODEL_RPN_POST_NMS_TOPK_TRAIN" : options.MODEL_RPN_POST_NMS_TOPK_TRAIN,
        "MODEL_RPN_POST_NMS_TOPK_TEST" : options.MODEL_RPN_POST_NMS_TOPK_TEST,
        "TEST_DETECTIONS_PER_IMAGE" : options.TEST_DETECTIONS_PER_IMAGE,
        "ANCHOR_GENERATOR_EXPECTED_SHAPES" : options.ANCHOR_GENERATOR_EXPECTED_SHAPES,
        "CLASSES_NAMES" : options.CLASSES_NAMES,
        "RADAR_NMS" : options.RADAR_NMS
    }

    train_dcs = reg_dataset(name_train_dataset, train_imgs_folder, train_annotation_path)
    if (name_test_dataset  != "") and (test_imgs_folder != "") and (test_annotation_path):
        test_dcs = reg_dataset(name_test_dataset, test_imgs_folder, test_annotation_path)

    # base_cfg_path = "configs/Base-RCNN-FPN.yaml"
    cfg = set_cfg_params(params, base_cfg_path)

    # write config
    cfg_name = "sartectron_config.yaml"
    cfg_name = write_cfg(cfg, outp_weights_path + "/" + cfg_name)

    # Validation set is exist
    if (name_test_dataset  != "") and (test_imgs_folder != "") and (test_annotation_path):
        trainer = CustomTrainerAndVal(cfg)
        if (options.MODEL_WEIGHTS[-4:] == ".pth"):
            cfg.MODEL_WEIGHTS = options.MODEL_WEIGHTS
            trainer.resume_or_load(resume = True)
        else:
            trainer.resume_or_load(resume = False)
        trainer.train()
    else:
        trainer = CustomTrainerNoVal(cfg)
        if (options.MODEL_WEIGHTS[-4:] == ".pth"):
            cfg.MODEL_WEIGHTS = options.MODEL_WEIGHTS
            trainer.resume_or_load(resume = True)
        else:
            trainer.resume_or_load(resume = False)
        trainer.train()
    return


def detecting_from_dir(testing_dir, saving_dir, cfg):
    predictor = CustomPredictor(cfg)
    imgs = []
    for i in os.listdir(testing_dir):
        if (i[-3:] == "bmp") or (i[-3:] == "tif"):
            imgs.append(i)
    print(f"Test images number is {len(imgs)}", flush=True)

    i = 1
    start_time = time.time()
    thing_colors = [tuple(np.random.choice(range(256), size=3)) for _ in cfg.DATASETS.CLASSES_NAMES]
    metadata = Metadata()
    metadata.set(thing_classes = cfg.DATASETS.CLASSES_NAMES, thing_colors = thing_colors, evaluator_type = "coco")
   
    shape_json = {}
    for img_name in imgs:
        img_path = testing_dir + "/" + img_name
        image = utils.read_image(img_path)
        output = predictor(image)

        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        v = Visualizer(
            image[:, :, ::-1],
            metadata = metadata,
            # scale=0.8,
            instance_mode=ColorMode.SEGMENTATION,
        )
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        num_founded = len(output["instances"].pred_boxes)
        print(f"Image #{i} : founded {num_founded} objects", flush = True)
        if (img_name[:6] == "new_H_"):
            if (num_founded != 0):
                search = re.search('new_H_(.*)_W_(.*).tif', img_name)
                H = int(search.group(1))
                W = int(search.group(2))

                temp = []
                for ind, coordinates in enumerate(output["instances"].pred_boxes.to("cpu")):
                    class_index = output["instances"].pred_classes[ind]
                    class_name = metadata.thing_classes[class_index]
                    abs_coord = [
                                    round(coordinates[0].item()) + W, round(coordinates[1].item()) + H,
                                    round(coordinates[2].item()) + W, round(coordinates[3].item()) + H
                                ]
                    prob = output["instances"].scores[ind].item()
                    print(f"    Object #{ind} : Class '{class_name}', prob = {prob}", flush = True)
                    temp.append({"bbox" : abs_coord, "class_name" : class_name, "prob" : prob})
                shape_json[img_path] = temp
        else:
            if (num_founded != 0):
                for ind, coordinates in enumerate(output["instances"].pred_boxes.to("cpu")):
                    class_index = output["instances"].pred_classes[ind]
                    class_name = metadata.thing_classes[class_index]
                    abs_coord = [
                                    round(coordinates[0].item()), round(coordinates[1].item()),
                                    round(coordinates[2].item()), round(coordinates[3].item())
                                ]
                    prob = output["instances"].scores[ind].item()
                    print(f"    Object #{ind} : Class '{class_name}', prob = {prob}", flush = True)
            cv2.imwrite(saving_dir + "\\" + img_name, v.get_image()[:, :, ::-1])

        # if (len(output["instances"].pred_boxes) != 0):
        #     cv2.imwrite(saving_dir + "\\" + img_name, v.get_image()[:, :, ::-1])
        #     # cv2.imshow("images", v.get_image()[:, :, ::-1])
        #     # cv2.waitKey(0)
        #     print(f"Image {i} / {len(imgs)} is done.", flush=True)

        i += 1

    if len(shape_json) != 0:
        shape_json = non_maximum_suppression(shape_json, cfg.MODEL.RADAR_NMS, 5, 5)
        res_shape = saving_dir + "/" + "detection_results.shp"
        with open(res_shape, "w") as f:
            json.dump(shape_json, f, indent=4)

    print("\n--- Time spend for detection: %s seconds ---" % (time.time() - start_time), flush=True)
    return


def TestBegin(options):
    if (options.yaml_file[-5:] != ".yaml") or (options.testing_folder == "") or (options.saving_folder == ""):
        print("YAML file invalid or testing or saving folders are empty.", flush = True)
        return

    cfg = get_cfg()
    cfg.merge_from_file(options.yaml_file)

    if (cfg.OUTPUT_DIR == "") or (cfg.MODEL.WEIGHTS == "") or (len(cfg.MODEL.ANCHOR_GENERATOR.EXPECTED_SHAPES) != len(cfg.DATASETS.CLASSES_NAMES)):
        print("Output dir is empty or no model weights file or shapes is broken.", flush = True)
        return

    detecting_from_dir(options.testing_folder, options.saving_folder, cfg)
    return


def foo_callback_dgt(option, opt, value, parser):
    digits_lst = [float(i.strip()) for i in value.split(",")]
    res = [[digits_lst[i], digits_lst[i + 1]] for i in range (0, len(digits_lst), 2)]
    setattr(parser.values, option.dest, res)


def foo_callback_lst(option, opt, value, parser):
    res = value.split(",")
    # print(res)
    setattr(parser.values, option.dest, res)


def parse_params():
    sys.setrecursionlimit(40000)
    parser = OptionParser()
    
    # Train mode
    parser.add_option("--train", "--train_network", dest="train_network", help="Set SARTectron in training mode.", action="store_true", default=False)

    parser.add_option("--tr_f", "--train_folder", dest="train_folder", help="Path to folder with training data.")
    parser.add_option("--tst_f", "--test_folder", dest="test_folder", help="Path to folder with testing data.")
    parser.add_option("--tr_ann_p", "--train_ann_path", dest="train_ann_path", help="Path to train annotation json-file.")
    parser.add_option("--tst_ann_p", "--test_ann_path", dest="test_ann_path", help="Path to test annotation json-file.")

    parser.add_option("--NAME_TR_DATASET", "--NAME_OF_TRAIN_DATASET", type="string", dest="NAME_OF_TRAIN_DATASET", help="Name of training dataset.")
    parser.add_option("--NAME_TEST_DATASET", "--NAME_OF_TEST_DATASET", type="string", dest="NAME_OF_TEST_DATASET", help="Name of testing dataset.")
    parser.add_option("--OUTPUT_DIR", "--OUTPUT_DIR", dest="OUTPUT_DIR", help="Path to folder where weights will be saved.")
    parser.add_option("--BASE_CFG_PATH", "--BASE_CFG_PATH", dest="BASE_CFG_PATH", help="BASE_CFG_PATH", default = "")

    parser.add_option("--M_ROI_HEADS_NUM_CLASSES", "--MODEL_ROI_HEADS_NUM_CLASSES", type = "int", dest="MODEL_ROI_HEADS_NUM_CLASSES", help="MODEL_ROI_HEADS_NUM_CLASSES.")
    parser.add_option("--CLASSES_NAMES", type="string", dest="CLASSES_NAMES", help="CLASSES_NAMES", action="callback", callback=foo_callback_lst)
    parser.add_option("--ANCHOR_GENERATOR_EXPECTED_SHAPES", type="string", dest="ANCHOR_GENERATOR_EXPECTED_SHAPES", help="ANCHOR_GENERATOR_EXPECTED_SHAPES", action="callback", callback=foo_callback_dgt)
    
    parser.add_option("--M_WEIGHTS", "--MODEL_WEIGHTS", dest="MODEL_WEIGHTS", help="MODEL_WEIGHTS", default = "")
    parser.add_option("--TEST_EVAL_PERIOD", "--TEST_EVAL_PERIOD", type = "int", dest="TEST_EVAL_PERIOD", help="TEST_EVAL_PERIOD", default = 100)
    parser.add_option("--M_DEVICE", "--MODEL_DEVICE", dest="MODEL_DEVICE", help="MODEL_DEVICE", default = "cpu")
    parser.add_option("--INP_MIN_SIZE_TR", "--INPUT_MIN_SIZE_TRAIN", type = "int", dest="INPUT_MIN_SIZE_TRAIN", help="INPUT_MIN_SIZE_TRAIN", default = 1000)
    parser.add_option("--INP_MAX_SIZE_TR", "--INPUT_MAX_SIZE_TRAIN", type = "int", dest="INPUT_MAX_SIZE_TRAIN", help="INPUT_MAX_SIZE_TRAIN", default = 1000)
    parser.add_option("--INP_MIN_SIZE_TEST", "--INPUT_MIN_SIZE_TEST", type = "int", dest="INPUT_MIN_SIZE_TEST", help="INPUT_MIN_SIZE_TEST", default = 1000)
    parser.add_option("--INP_MAX_SIZE_TEST", "--INPUT_MAX_SIZE_TEST", type = "int", dest="INPUT_MAX_SIZE_TEST", help="INPUT_MAX_SIZE_TEST", default = 1000)
    parser.add_option("--DLOADER_NUM_WORKERS", "--DATALOADER_NUM_WORKERS", type = "int", dest="DATALOADER_NUM_WORKERS", help="DATALOADER_NUM_WORKERS", default = 0)
    parser.add_option("-v", "--DATALOADER_FILTER_EMPTY_ANNOTATIONS", dest="DATALOADER_FILTER_EMPTY_ANNOTATIONS", help="DATALOADER_FILTER_EMPTY_ANNOTATIONS",  default = False)
    parser.add_option("--M_RPN_BATCH_SIZE_PER_IMAGE", "--MODEL_RPN_BATCH_SIZE_PER_IMAGE", type = "int", dest="MODEL_RPN_BATCH_SIZE_PER_IMAGE", help="MODEL_RPN_BATCH_SIZE_PER_IMAGE", default = 1024)
    parser.add_option("--M_ROI_HEADS_BATCH_SIZE_PER_IMAGE", "--MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE", type = "int", dest="MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE", help="MODEL_ROI_HEADS_BATCH_SIZE_PER_IMAGE", default = 2048)
    parser.add_option("--M_RPN_POS_FRAC", "--MODEL_RPN_POS_FRACTION", type = "float", dest="MODEL_RPN_POS_FRACTION", help="M_RPN_POS_FRACTION", default = 0.5)
    parser.add_option("--M_ROI_HEADS_POS_FRAC", "--MODEL_ROI_HEADS_POS_FRACTION", type = "float", dest="MODEL_ROI_HEADS_POS_FRACTION", help="MODEL_ROI_HEADS_POS_FRACTION", default = 0.25)
    parser.add_option("--SOLV_IMS_PER_BATCH", "--SOLVER_IMS_PER_BATCH", type = "int", dest="SOLVER_IMS_PER_BATCH", help="SOLVER_IMS_PER_BATCH", default = 1)
    parser.add_option("--SOLV_CHECKPOINT_PERIOD", "--SOLVER_CHECKPOINT_PERIOD", type = "int", dest="SOLVER_CHECKPOINT_PERIOD", help="SOLVER_CHECKPOINT_PERIOD", default = 500)
    parser.add_option("--SOLV_BASE_LR", "--SOLVER_BASE_LR", type = "float", dest="SOLVER_BASE_LR", help="SOLVER_BASE_LR", default = 0.001)
    parser.add_option("--SOLV_MAX_ITER", "--SOLVER_MAX_ITER", type = "int", dest="SOLVER_MAX_ITER", help="SOLVER_MAX_ITER", default = 100000)
    parser.add_option("--M_RPN_PRE_NMS_TOPK_TR", "--MODEL_RPN_PRE_NMS_TOPK_TRAIN", type = "int", dest="MODEL_RPN_PRE_NMS_TOPK_TRAIN", help="MODEL_RPN_PRE_NMS_TOPK_TRAIN", default = 3000)
    parser.add_option("--M_RPN_PRE_NMS_TOPK_TEST", "--MODEL_RPN_PRE_NMS_TOPK_TEST", type = "int", dest="MODEL_RPN_PRE_NMS_TOPK_TEST", help="MODEL_RPN_PRE_NMS_TOPK_TEST", default = 2000)
    parser.add_option("--M_RPN_POST_NMS_TOPK_TR", "--MODEL_RPN_POST_NMS_TOPK_TRAIN", type = "int", dest="MODEL_RPN_POST_NMS_TOPK_TRAIN", help="MODEL_RPN_POST_NMS_TOPK_TRAIN", default = 3000)
    parser.add_option("--M_RPN_POST_NMS_TOPK_TEST", "--MODEL_RPN_POST_NMS_TOPK_TEST", type = "int", dest="MODEL_RPN_POST_NMS_TOPK_TEST", help="MODEL_RPN_POST_NMS_TOPK_TEST", default = 2000)
    parser.add_option("--TEST_DETECTIONS_PER_IMAGE", "--TEST_DETECTIONS_PER_IMAGE", type = "int", dest="TEST_DETECTIONS_PER_IMAGE", help="TEST_DETECTIONS_PER_IMAGE", default = 200)
    parser.add_option("--RDR_NMS", "--RADAR_NMS", type = "float", dest="RADAR_NMS", help="RADAR_NMS", default = 0.5)

    # Test mode
    parser.add_option("--test", "--test_from_dir", dest="test_from_dir", help="Set SARTectron in interference mode.", action="store_true", default=False)

    parser.add_option("--yaml", "--yaml_file", dest="yaml_file", help="Full path to yaml file.", default="")
    parser.add_option("--testing_f", "--testing_folder", dest="testing_folder", help="Path to folder with images to detect objects on.", default="")
    parser.add_option("--saving_f", "--saving_folder", dest="saving_folder", help="Path to folder, where you want to save results of detection.", default="")
    (options, args) = parser.parse_args()

    if not options.test_from_dir and not options.train_network:
        parser.error("Set SARTectron on Training or Testing mode with flags '--train_network' or '--test_from_dir' first.")
    elif options.test_from_dir and options.train_network:
        parser.error("Choose one mode for neural network: '--train_network' OR '--test_from_dir'.")

    # Test
    if options.test_from_dir and not options.train_network:
        TestBegin(options)
    # Train
    elif not options.test_from_dir and options.train_network:
        TrainBegin(options)

    # python SARTectron.py --train_folder --device cuda --yaml_file D:/Train_Test_Detectron2/Dataset_Real_28_09_2021/weights/sartectron_config.yaml --model_folder D:/Train_Test_Detectron2/Dataset_Real_28_09_2021/weights --model_name model_0056999.pth --name_of_dataset coco_Planes_detection_Train --train_train_imgs_folder D:/Train_Test_Detectron2/Dataset_Real_28_09_2021 --annotation_file D:/Train_Test_Detectron2/Dataset_Real_28_09_2021/Train_Data.json --testing_folder D:/Train_Test_Detectron2/Dataset_Real_28_09_2021 --saving_folder D:/Train_Test_Detectron2/Dataset_Real_28_09_2021/weights/results_detection


    # {
        # train_imgs_folder = options.train_folder
        # test_imgs_folder = options.test_folder
        # train_annotation_path = options.train_ann_path
        # test_annotation_path = options.test_ann_path

        # name_train_dataset = options.NAME_OF_TRAIN_DATASET
        # name_test_dataset = options.NAME_OF_TEST_DATASET
        # outp_weights_path = options.OUTPUT_DIR
        
        # train_begin(
        #     name_train_dataset, name_test_dataset,
        #     train_imgs_folder, test_imgs_folder,
        #     train_annotation_path, test_annotation_path,
        #     outp_weights_path
        #     )
    # }


def main():
    parse_params()
    
    # config_file = "D:/Train_Test_Detectron2/del_OUTPUT/sartectron_config.yaml"
    # model_name = "model_0000099.pth"
    # testing_dir = "D:/Train_Test_Detectron2/Dataset_10k/Test"
    # saving_dir = "D:/Train_Test_Detectron2/del_OUTPUT/results"

    # train_begin("coco_Planes_detection_Train",
    #     "coco_Planes_detection_Test",
    #     "D:/Train_Test_Detectron2/Dataset_10k/Train",
    #     "D:/Train_Test_Detectron2/Dataset_10k/Test",
    #     "D:/Train_Test_Detectron2/Dataset_10k/Train/Train_Data_with_no_objects.json",
    #     "D:/Train_Test_Detectron2/Dataset_10k/Test/Test_Data_REAL.json",
    #     "D:/Train_Test_Detectron2/res_learning_16_07"
    # )
    # train_begin("coco_Planes_detection_Train",
    #     "coco_Planes_detection_Test",
    #     "D:/Train_Test_Detectron2/Dataset_Real_28_09_2021",
    #     "D:/Train_Test_Detectron2/Dataset_10k/Test",
    #     "D:/Train_Test_Detectron2/Dataset_Real_28_09_2021/Train_Data.json",
    #     "D:/Train_Test_Detectron2/Dataset_10k/Test/Test_Data_REAL.json",
    #     "D:/Train_Test_Detectron2/Dataset_Real_28_09_2021/weights"
    # )


if __name__ == "__main__":
    main()
