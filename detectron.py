from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

cfg = get_cfg()

cfg.MODEL.DEVICE = "cpu"
# load the pre trained model from Detectron2 model zoo
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
# set confidence threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
# load model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
# create the predictor for pose estimation using the config
predictor = DefaultPredictor(cfg)

# Load and preprocess the input image
image_path = 'sample_imgs/01567585-a.jpeg'

async def run_detectron(img):
    # image = cv2.imread(image_path)
    # im2 = cv2.imread(image_path)
    image = np.array(img)
    height, width = image.shape[:2]

    # Perform pose estimation
    outputs = predictor(image)
    keypoints = outputs['instances'].pred_keypoints.to('cpu')[0]
    print(keypoints)
    # Visualize the predicted keypoints on the image
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image = out.get_image()[:, :, ::-1]
    metad = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    k_dct = {}
    for i,j in zip(metad.keypoint_names,keypoints):

        if j[2].item() > 0:
            x = int(j[0].item() * 1)
            y = int(j[1].item() * 1)
            k_dct[i] = (x,y)
        else:
            k_dct[i] = None
        # cv2.circle(im2,(x,y),4,(0,255,255),-1)
    print(k_dct)
    return k_dct
# Display the output image
# cv2.imshow("Pose Estimation", im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
