from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import asyncio


app = FastAPI()


@app.get("/")
def home():
    return {"Hello": "Welcome to pose estimation api.....!"}


@app.post("/send_img")
async def get_img(file: UploadFile):
    await asyncio.sleep(10)
    img = Image.open(BytesIO(await file.read()))
    print(img.size)
    return {"message": f"Image {file.filename} {img.size} uploaded succesfully..."}
