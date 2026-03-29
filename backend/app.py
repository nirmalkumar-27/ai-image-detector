from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from model import AIDetector
from utils import read_image
import os

app = FastAPI()

detector = AIDetector()

# serve frontend
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("../frontend/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

        if ai_prob > 0.6:
            label = "AI Generated"
            analysis = "Synthetic patterns detected"
        else:
            label = "Real Image"
            analysis = "Natural image"

        return {
            "label": label,
            "confidence": round(ai_prob * 100, 2),
            "analysis": analysis
        }

    except Exception as e:
        return {"error": str(e)}