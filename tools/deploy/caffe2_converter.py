#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import onnx

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import Caffe2Tracer, add_export_config
from detectron2.modeling import build_model
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.logger import setup_logger


def reg_dataset(name_of_dataset, imgs_root, f_path_annotation):
    # register_coco_instances(name_of_dataset, {}, f_path_annotation,imgs_root)
    load_coco_json(f_path_annotation, imgs_root, dataset_name=name_of_dataset)
    DatasetCatalog.register(name_of_dataset, lambda: load_coco_json(f_path_annotation, imgs_root, name_of_dataset))
    return name_of_dataset

def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = add_export_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if cfg.MODEL.DEVICE != "cpu":
        assert TORCH_VERSION >= (1, 5), "PyTorch>=1.5 required for GPU conversion!"
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model using caffe2 tracing.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="caffe2",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    os.makedirs(args.output, exist_ok=True)

    # CHECK
    # python caffe2_converter.py
    # --config-file C:/Users/savchenko.bs/Desktop/Demonstration/Results_on_testing_dataset/Weights_Detectron2/detectron2_config.yaml
    # --output C:/Users/savchenko.bs/Desktop
    # MODEL.WEIGHTS C:/Users/savchenko.bs/Desktop/Demonstration/Results_on_testing_dataset/Weights_Detectron2/model_0010749.pth
    # MODEL.DEVICE cpu


    name_of_dataset = "coco_Planes_detection_Train"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k"
    f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/Dataset_10k/Train_Data.json"
    reg_dataset(name_of_dataset, imgs_root, f_path_annotation)

    name_of_test_dataset = "coco_Planes_detection_Test"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF"
    test_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/RDP_1000_1000/TIFF/Test_Data_REAL.json"
    reg_dataset(name_of_test_dataset, imgs_root, test_path_annotation)

    thing_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(thing_colors=thing_colors)
    stuff_colors = [(28, 3, 252), (3, 252, 98), (252, 3, 3), (169, 3, 252)]
    MetadataCatalog.get(name_of_dataset).set(stuff_colors=stuff_colors)

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)

    # get a sample data
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    first_batch = next(iter(data_loader))

    # convert and save caffe2 model
    tracer = Caffe2Tracer(cfg, torch_model, first_batch)
    if args.format == "caffe2":
        caffe2_model = tracer.export_caffe2()
        caffe2_model.save_protobuf(args.output)
        # draw the caffe2 graph
        caffe2_model.save_graph(os.path.join(args.output, "model.svg"), inputs=first_batch)
    elif args.format == "onnx":
        onnx_model = tracer.export_onnx()
        onnx.save(onnx_model, os.path.join(args.output, "model.onnx"))
    elif args.format == "torchscript":
        script_model = tracer.export_torchscript()
        script_model.save(os.path.join(args.output, "model.ts"))

        # Recursively print IR of all modules
        with open(os.path.join(args.output, "model_ts_IR.txt"), "w") as f:
            try:
                f.write(script_model._actual_script_module._c.dump_to_str(True, False, False))
            except AttributeError:
                pass
        # Print IR of the entire graph (all submodules inlined)
        with open(os.path.join(args.output, "model_ts_IR_inlined.txt"), "w") as f:
            f.write(str(script_model.inlined_graph))
        # Print the model structure in pytorch style
        with open(os.path.join(args.output, "model.txt"), "w") as f:
            f.write(str(script_model))

    # run evaluation with the converted model
    if args.run_eval:
        assert args.format == "caffe2", "Python inference in other format is not yet supported."
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, cfg, True, args.output)
        metrics = inference_on_dataset(caffe2_model, data_loader, evaluator)
        print_csv_format(metrics)
