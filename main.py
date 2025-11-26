# main.py
import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import torch

# Make PyTorch lighter on small instances
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")  # nano model

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model from {MODEL_PATH}: {e}")

app = FastAPI(title="YOLO Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten later if you want
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

    # 2. Read image
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image file.")

    orig_w, orig_h = img.size

    # 3. Downscale big images to save RAM
    MAX_SIZE = 1024
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    resized_w, resized_h = img.size

    # 4. Run YOLO
    try:
        results = model.predict(
            img,
            imgsz=min(MAX_SIZE, 640),
            conf=0.25,
            verbose=False
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # 5. Build detections in BOTH formats:
    #    - boxes/scores/class_ids  (old frontend)
    #    - detections[]            (richer objects)
    boxes_list = []
    scores_list = []
    class_ids_list = []
    detections = []

    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        names = r.names

        if boxes is not None:
            for box in boxes:
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

    # 6. Response: compatible with your old frontend,
    #    plus extra metadata if you want to use it later.
    return {
        "boxes": boxes_list,
        "scores": scores_list,
        "class_ids": class_ids_list,
        "detections": detections,
        "original_width": orig_w,
        "original_height": orig_h,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "scale_x": (orig_w / resized_w) if resized_w else 1.0,
        "scale_y": (orig_h / resized_h) if resized_h else 1.0,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
