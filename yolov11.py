from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import io
import torchvision
import tempfile
import time
from torchvision.ops import nms

app = FastAPI()
# Load YOLOv11n model on GPU
model = YOLO("yolo11n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())

def annotate_frame(frame, results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{int(cls)}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
        for box, score, cls in zip(boxes, scores, classes):
            detections.append({
                "box": box.tolist(),
                "score": float(score),
                "class": int(cls)
            })

    return {"results": detections}

@app.post("/predict_video/")
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()
    # Save uploaded video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as in_tmp:
        in_tmp.write(contents)
        in_tmp_path = in_tmp.name

    cap = cv2.VideoCapture(in_tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_tmp:
        out_tmp_path = out_tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_tmp_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = annotate_frame(frame, results)
        out_vid.write(annotated)

    cap.release()
    out_vid.release()

    # Return the annotated video
    def iterfile():
        with open(out_tmp_path, "rb") as f:
            yield from f

    return StreamingResponse(iterfile(), media_type="video/mp4")