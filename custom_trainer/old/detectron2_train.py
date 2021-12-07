from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog

import cv2
from detectron2.utils.visualizer import Visualizer


def reg_dataset(name_of_dataset, imgs_root, f_path_annotation):
    # register_coco_instances(name_of_dataset, {}, f_path_annotation,imgs_root)
    load_coco_json(f_path_annotation, imgs_root, dataset_name=name_of_dataset)
    DatasetCatalog.register(
        name_of_dataset, lambda: load_coco_json(
            f_path_annotation, imgs_root, name_of_dataset
            )
        )
    return name_of_dataset


def create_cfg(weights_root, name_of_dataset):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = weights_root

    cfg.DATASETS.TRAIN = (name_of_dataset,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset

    cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

    # cfg.INPUT.RANDOM_FLIP = "horizontal"
    # cfg.DATALOADER.NUM_WORKERS = 8
    # cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.CHECKPOINT_PERIOD = 100

    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 48, 64, 128, 256]]
    # P3_C, KC_135, C_5, B_52
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
        [0.5, 1.0, 1.8, 2.17, 2.3]
        ]
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


def main():
    # registration dataset
    name_of_dataset = "Planes_detection_Train"
    imgs_root = "C:/Users/savchenko.bs/Desktop/new_placement/Detectron2_dataset"
    f_path_annotation = "C:/Users/savchenko.bs/Desktop/new_placement/Detectron2_dataset/Train_Data_FULL.json"

    name_of_dataset = reg_dataset(
        name_of_dataset, imgs_root, f_path_annotation
        )

    # MetadataCatalog.get(name_of_dataset).thing_classes = [
    # "B_52", "C_5", "KC_135", "P_3C"
    # ]
    print(MetadataCatalog.get(name_of_dataset))

    # visualize_img(name_of_dataset, 25654)

    # config params function
    weights_root = "C:/Users/savchenko.bs/Desktop/new_placement/detectron2/weights/results_learning_new"
    cfg = create_cfg(weights_root, name_of_dataset)

    # write weights specific from config
    # name_model = write_weights_from_cfg(cfg, weights_root,"detectron2_model")

    cfg_name = "detectron2_config.yaml"
    cfg_name = write_cfg(cfg, weights_root + "/" + cfg_name)

    cfg_from_file = get_cfg()
    cfg_from_file.merge_from_file(cfg_name)

    trainer = DefaultTrainer(cfg_from_file)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
