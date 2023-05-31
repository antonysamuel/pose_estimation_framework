from ultralytics import YOLO
from PIL import Image


img = Image.open('sample_imgs/01567585-a.jpeg')
# Load a model

async def run_yolo(img):
    model = YOLO('yolov8n-pose.pt')  # load an official model

    results = model(img,save=False)  # predict on an image
    key_points = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    k_dct = {}
    print(results[0].keypoints)
    for i,j in zip(key_points,results[0].keypoints[0]):
        if j[2].item() > .3:
            x = int(j[0].item())
            y = int(j[1].item())
            k_dct[i] = (x,y)
        else:
            k_dct[i] = None
    print(k_dct)
    return k_dct

