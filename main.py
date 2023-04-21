from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO(r"C:\Users\grave\anaconda3\envs\brownSort\ExtraEnv\yolov8n.pt")

model.predict(source="0", show=True, conf=0.5) # accepts all formats - blabla....  