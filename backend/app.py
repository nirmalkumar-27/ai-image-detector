from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import AIDetector
from utils import read_image

import base64
import cv2
import numpy as np

app = FastAPI()

detector = AIDetector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_image(contents)

        real_prob = detector.predict(image)
        ai_prob = 1 - real_prob

        # decision logic
        if ai_prob > 0.7:
            label = "AI Generated"
            analysis = "Strong synthetic patterns detected"
        elif ai_prob > 0.5:
            label = "Possibly AI Generated"
            analysis = "Some artificial characteristics detected"
        else:
            label = "Real Image"
            analysis = "Natural image properties detected"

        # heatmap
        cam = detector.generate_heatmap(image)
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        _, buffer = cv2.imencode(".jpg", cam)
        heatmap_base64 = base64.b64encode(buffer).decode()

        return {
            "label": label,
            "confidence": round(ai_prob * 100, 2),
            "analysis": analysis,
            "heatmap": heatmap_base64
        }

    except Exception as e:
        return {"error": str(e)}