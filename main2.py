# This version of the program will detect classes and gives a variable "center" that gives the center of the class.
# Next version should calculate how long each pixel corresponds to in the real world when placing the camera a given length 
# from the conveyor belt.

from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO(r"C:\Users\grave\anaconda3\envs\brownSort\ExtraEnv\yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)

    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            print("Bounding Box Coordinates: "+str(b))         
            c = box.cls
            print(model.names[int(c)])
            annotator.box_label(b, model.names[int(c)])
            center = [(float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2]
            print(str(model.names[int(c)])+" is located with center at: "+str(center))

          
    frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', frame)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()