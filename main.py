# main.py
import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch

# Keep PyTorch light on small CPUs
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Which model to use (override via env if you want)
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model from {MODEL_PATH}: {e}")

app = FastAPI(title="YOLO Backend", version="1.0.0")

# CORS – allow your frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # you can restrict to your domain later
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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # 2. Read and decode image
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    orig_w, orig_h = img.size

    # 3. Run YOLO (let YOLO handle internal resizing to imgsz=640)
    try:
        results = model.predict(
            img,
            imgsz=640,      # standard YOLO size; change if you like
            conf=0.25,      # confidence threshold
            verbose=False
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 4. Build detections in both formats:
    #    - boxes/scores/class_ids  (used by your current frontend)
    #    - detections[]            (richer object list if needed later)
    boxes_list = []
    scores_list = []
    class_ids_list = []
    detections = []

    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        names = r.names  # class id -> name mapping

        if boxes is not None:
            for box in boxes:
                # xyxy are already in original-image coordinates for this img
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = (
                    names.get(cls_id, str(cls_id))
                    if isinstance(names, dict)
                    else str(cls_id)
                )

                boxes_list.append(xyxy)
                scores_list.append(conf)
                class_ids_list.append(cls_id)

                detections.append(
                    {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )

    # 5. JSON response – fully compatible with your yolo.js
    return {
        "boxes": boxes_list,
        "scores": scores_list,
        "class_ids": class_ids_list,
        "detections": detections,
        "original_width": orig_w,
        "original_height": orig_h,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
