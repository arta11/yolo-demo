# main.py
import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch

# ================
# Torch / YOLO setup
# ================
# Make PyTorch lighter on tiny servers
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")  # keep it nano

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    # Fail fast if model can't be loaded
    raise RuntimeError(f"Failed to load YOLO model from {MODEL_PATH}: {e}")

# ================
# FastAPI app
# ================
app = FastAPI(title="YOLO Backend", version="1.0.0")

# Allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production you can restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "YOLO backend is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # 2. Read file into memory
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    # Original size (what the phone actually took)
    orig_w, orig_h = img.size

    # 3. Downscale large images to save RAM (critical on 512MB servers)
    #    This keeps aspect ratio, max 1024px on any side.
    MAX_SIZE = 1024
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    resized_w, resized_h = img.size

    # 4. Run YOLO (on resized image)
    try:
        results = model.predict(
            img,
            imgsz=min(MAX_SIZE, 640),  # inference size
            conf=0.25,
            verbose=False
        )
    except RuntimeError as e:
        # Typical when running out of memory
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    detections = []
    # We expect one image, so take results[0]
    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        names = r.names  # class id -> name mapping

        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                detections.append({
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

    # 5. Return sizes + boxes
    #    Frontend can use resized_* for canvas drawing,
    #    or use scale factors to map to original phone resolution.
    return {
        "original_width": orig_w,
        "original_height": orig_h,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "scale_x": orig_w / resized_w,
        "scale_y": orig_h / resized_h,
        "detections": detections,
    }


# For local testing:
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
