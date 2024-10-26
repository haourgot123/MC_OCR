import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import numpy as np
import os
from datetime import datetime
from roboflow import Roboflow
# from config.config import  roboflow_config

#Tải file config đã được train 
class DetectronSegmentation():
    def __init__(self, config_path):
        self.cfg = get_cfg()
        self.config_path = config_path
    # Dowload dataset for train_segmentation
    # def load_dataset(self):
    #     rf = Roboflow(api_key = roboflow_config['api_key'])
    #     project =rf.workspace(roboflow_config['workspace']).project(roboflow_config['project_name'])
    #     version = project.version(roboflow_config['version'])
    #     self.dataset = version.dowload(roboflow_config['type'])
    #     return self.dataset
    def register_dataset(self):
        # TRAIN SET
        self.DATA_SET_NAME = self.dataset.name.replace(" ", "-")
        self.ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
        self.TRAIN_DATA_SET_NAME = f"{self.DATA_SET_NAME}-train"
        self.TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(self.dataset.location, "train")
        self.TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(self.dataset.location, "train", self.ANNOTATIONS_FILE_NAME)
        # TEST SET
        self.TEST_DATA_SET_NAME = f"{self.DATA_SET_NAME}-test"
        self.TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(self.dataset.location, "test")
        self.TEST_DATA_SET_ANN_FILE_PATH = os.path.join(self.dataset.location, "test", self.ANNOTATIONS_FILE_NAME)
        # VALID SET
        self.VALID_DATA_SET_NAME = f"{self.DATA_SET_NAME}-valid"
        self.VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(self.dataset.location, "valid")
        self.VALID_DATA_SET_ANN_FILE_PATH = os.path.join(self.dataset.location, "valid", self.ANNOTATIONS_FILE_NAME)
        # REGISTER TRAIN
        register_coco_instances(
            name=self.TRAIN_DATA_SET_NAME,
            metadata={},
            json_file=self.TRAIN_DATA_SET_ANN_FILE_PATH,
            image_root=self.TRAIN_DATA_SET_IMAGES_DIR_PATH
        )
        # REGISTER TEST
        register_coco_instances(
            name=self.TEST_DATA_SET_NAME,
            metadata={},
            json_file=self.TEST_DATA_SET_ANN_FILE_PATH,
            image_root=self.TEST_DATA_SET_IMAGES_DIR_PATH
        )
        # REGISTER VALID
        register_coco_instances(
            name=self.VALID_DATA_SET_NAME,
            metadata={},
            json_file=self.VALID_DATA_SET_ANN_FILE_PATH,
            image_root=self.VALID_DATA_SET_IMAGES_DIR_PATH
        )
    def visualize(self):
        metadata = MetadataCatalog.get(self.TRAIN_DATA_SET_NAME)
        dataset_train = DatasetCatalog.get(self.TRAIN_DATA_SET_NAME)
        dataset_entry = dataset_train[0]
        image = cv2.imread(dataset_entry['file_name'])
        visualizer = Visualizer(
            image[:, :, ::-1],
            metadata = metadata,
            scale = 0.8,
            instance_mode = ColorMode.IMAGE_BW
        )
        out  = visualizer.draw_dataset_dict(dataset_entry)
    def configuration(self):
        self.ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
        self.CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{self.ARCHITECTURE}.yaml"
        self.MAX_ITER = 2000
        self.EVAL_PERIOD = 200
        self.BASE_LR = 0.001
        self.NUM_CLASSES = 3
        self.OUTPUT_DIR_PATH = os.path.join(
            self.DATA_SET_NAME,
            self.ARCHITECTURE,
            datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        )
        os.makedirs(self.OUTPUT_DIR_PATH, exist_ok = True)
        self.cfg.merge_from_file(model_zoo.get_config_file(self.CONFIG_FILE_PATH))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.CONFIG_FILE_PATH)
        self.cfg.DATASETS.TRAIN = (self.TRAIN_DATA_SET_NAME,)
        self.cfg.DATASETS.TEST = (self.TEST_DATA_SET_NAME,)
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
        self.cfg.TEST.EVAL_PERIOD = self.EVAL_PERIOD
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.INPUT.MASK_FORMAT='bitmask'
        self.cfg.SOLVER.BASE_LR = self.BASE_LR
        self.cfg.SOLVER.MAX_ITER = self.MAX_ITER
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.NUM_CLASSES
        self.cfg.OUTPUT_DIR = self.OUTPUT_DIR_PATH
    def train(self):
        print('-------------------------LOADING DATASET-----------------------')
        self.load_dataset()
        print('-------------------------REGISTER DATASET----------------------')
        self.register_dataset()
        print('-------------------------CONFIGURATION-------------------------')
        self.configuration()
        print('--------------------------TRAINING-----------------------------')
        trainer = DefaultPredictor(self.cfg)
        trainer.resume_or_load(resume = False)
        trainer.train()
        print('--------------------------FINISH TRAINING----------------------')
    def evaluation(self):
        self.cfg.MODEL.WEIGHTS = '/content/Invoice_Segmentation/mask_rcnn_R_101_FPN_3x/2024-09-03-14-44-17/model_final.pth'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        predictor = DefaultPredictor(self.cfg)
        dataset_valid = DatasetCatalog.get(self.VALID_DATA_SET_NAME)
        result = []
        for d in dataset_valid:
            img = cv2.imread(d["file_name"])
            outputs = predictor(img)

            visualizer = Visualizer(
                img[:, :, ::-1],
                metadata=self.metadata,
                scale=0.8,
                instance_mode=ColorMode.IMAGE_BW
            )
            out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
            result.append(out)
        return result
    def predict(self, img):
        # Convert img to RGB
        trained_config = get_cfg()
        trained_config.merge_from_file(self.config_path)
        predictor = DefaultPredictor(trained_config)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        output  = predictor(img)
        masks = output['instances'].pred_masks
        masks = np.array(masks.to('cpu'))
        binary_masks = masks[0].astype(np.uint8) * 255
        binary_masks_3d = cv2.merge([binary_masks, binary_masks, binary_masks])
        img_res = cv2.bitwise_and(img, binary_masks_3d)
        img_res = cv2.cvtColor(img_res,cv2.COLOR_BGR2RGB)
        return img_res

# if __name__ == '__main__':
#     config_path = 'config.yml'
#     img_path = '../../../dataset/train_images/mcocr_public_145013acjke.jpg'
#     img = cv2.imread(img_path)
#     print(img.shape)
#     segment = DetectronSegmentation(config_path)
#     img_res = segment.predict(img)
#     print(img_res.shape)

