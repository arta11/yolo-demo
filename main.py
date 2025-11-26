from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Allow your frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict this to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model ONE TIME (fast)
model = YOLO("yolov8n.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Convert uploaded image to PIL
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(img)[0]

    detections = []
    for box, cls, conf in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.conf,
    ):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "class_name": model.names[int(cls)],
            "confidence": float(conf),
            "box": [x1, y1, x2, y2],
        })

    return {"detections": detections}
