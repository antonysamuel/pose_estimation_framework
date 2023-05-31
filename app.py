from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import asyncio
from mpipe import run_mediapipe
from mvnet import run_movenet
from detectron import run_detectron
from yolo import run_yolo
from posenetpytorch.posenetmain import run_posenet

app = FastAPI()


@app.get("/")
def home():
    return {"Hello": "Welcome to pose estimation api.....!"}


@app.post("/mediapipe")
async def get_img(file: UploadFile):
    # await asyncio.sleep(10)
    img = Image.open(BytesIO(await file.read()))
    keypoints = await run_mediapipe(img)
    if keypoints == None:
        return {'message': {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "mediapipe": "!NO KEYPOINTS FOUND..."}}
    return {"message": {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "mediapipe": keypoints}}


@app.post("/movenet")
async def get_img(file: UploadFile):
    # await asyncio.sleep(10)
    img = Image.open(BytesIO(await file.read()))
    keypoints = await run_movenet(img)
    if keypoints == None:
        return {'message': {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "movenet": "!NO KEYPOINTS FOUND..."}}
    return {"message": {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "movenet": keypoints}}

@app.post("/detectron")
async def get_img(file: UploadFile):
    # await asyncio.sleep(10)
    img = Image.open(BytesIO(await file.read()))
    keypoints = await run_detectron(img)
    if keypoints == None:
        return {'message': {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "detectron": "!NO KEYPOINTS FOUND..."}}
    return {"message": {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "detectron": keypoints}}


@app.post("/yolo")
async def get_img(file: UploadFile):
    # await asyncio.sleep(10)
    img = Image.open(BytesIO(await file.read()))
    keypoints = await run_yolo(img)
    if keypoints == None:
        return {'message': {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "yolo": "!NO KEYPOINTS FOUND..."}}
    return {"message": {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "yolo": keypoints}}

@app.post("/posenet")
async def get_img(file: UploadFile):
    # await asyncio.sleep(10)
    img = Image.open(BytesIO(await file.read()))
    keypoints = await run_posenet(img)
    # print("posenet kp",keypoints)
    if keypoints == None:
        return {'message': {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "posenet": "!NO KEYPOINTS FOUND..."}}
    return {"message": {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "posenet": keypoints}}



@app.post("/runall")
async def get_img(file: UploadFile):
    # await asyncio.sleep(10)
    img = Image.open(BytesIO(await file.read()))
    mpipe = await run_mediapipe(img)
    detectron = await run_detectron(img)
    posenet = await run_posenet(img)
    yolo = await run_yolo(img)
    movenet = await run_movenet(img)
    # print("posenet kp",keypoints)
    
    return {"message": {"status": f"Image {file.filename} {img.size} uploaded succesfully...", "mediapipe": mpipe, "detectron":detectron,"posenet":posenet,"yolo":yolo,"movenet":movenet }}
