from libs import *
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data='data.yaml', imgsz=640, epochs=50, batch=32, name='yolov8n_custom')
