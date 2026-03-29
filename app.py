from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from model import AIDetector
from utils import read_image

import base64
import cv2
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model once
detector = AIDetector()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_image(contents)

        # ---- Prediction ----
        real_prob = detector.predict(image)
        ai_prob = 1 - real_prob

        # ---- Decision ----
        if ai_prob > 0.7:
            label = "AI Generated"
            analysis = "Strong synthetic patterns detected"
        elif ai_prob > 0.5:
            label = "Possibly AI Generated"
            analysis = "Some artificial characteristics detected"
        else:
            label = "Real Image"
            analysis = "Natural image properties detected"

        # ---- Heatmap ----
        try:
            cam = detector.generate_heatmap(image)
            cam = np.uint8(255 * cam)
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

            _, buffer = cv2.imencode(".jpg", cam)
            heatmap_base64 = base64.b64encode(buffer).decode()
        except:
            heatmap_base64 = None  # fallback if heatmap fails

        return {
            "label": label,
            "confidence": round(ai_prob * 100, 2),
            "analysis": analysis,
            "heatmap": heatmap_base64
        }

    except Exception as e:
        return {"error": str(e)}


# 🔥 REQUIRED FOR HUGGING FACE / DEPLOYMENT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)