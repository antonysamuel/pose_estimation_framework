import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
kp_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

pose_detector = mp.solutions.pose.Pose()

image = Image.open("/home/sam/Codes/rapid/pose_framewrk_api/sample_imgs/000109154.jpg")
# im = cv2.imread("/home/sam/Codes/rapid/pose_framewrk_api/sample_imgs/000109154.jpg")
async def run_mediapipe(image):
    im_np = np.array(image)
    print(im_np.shape)
    results = pose_detector.process(im_np)
    if results.pose_landmarks == None:
        return None
    keyp = results.pose_landmarks.landmark
    keypoint_dict = {}

    # image_cv2 = cv2.imdecode(im_np, cv2.IMREAD_COLOR)

    for i in range(len(kp_names)):
        kp_name = kp_names[i]
        kp_coor = (int(keyp[i].x * im_np.shape[1]), int(keyp[i].y * im_np.shape[0]))
        keypoint_dict[kp_name] = kp_coor

    print(keypoint_dict)
    return keypoint_dict
