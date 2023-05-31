import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from PIL import Image

module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192

model = module.signatures['serving_default']
img_pth = 'sample_imgs/01567585-a.jpeg'
img_o = Image.open(img_pth)

async def run_movenet(img):
    image = tf.convert_to_tensor(img)
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.cast(tf.image.resize_with_pad(input_image, input_size, input_size),dtype=tf.int32)

    output = model(input_image)
    keypoints = output['output_0'].numpy()
    im_np = np.array(img_o)
    h,w,_ = im_np.shape
    box_size = max(h,w)
    print(h,w)
    print(len(keypoints[0][0]))
    key_points = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    kp_dct = {}
    i = 0
    for y,x,conf in keypoints[0][0]:
        if conf > 0.25:
            x = int(x * box_size - (box_size-w)/2)
            y = int(y * box_size - (box_size-h)/2)
            # x = int(x*w)
            # y = int(y*h)
            print(x,y,conf)
            kp_dct[key_points[i]] = (x,y) 
            i += 1
            # cv2.circle(img_oo,(x,y),4,(0,255,255),-1)
        else:
            kp_dct[key_points[i]] = None
            i += 1
    return kp_dct


# print(kp_dct)
# cv2.imshow('i',img_oo)
# cv2.waitKey(0)
# cv2.destroyAllWindows()