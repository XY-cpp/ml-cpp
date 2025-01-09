import os
import sys
import coremltools as ct
from ultralytics import YOLO

name = sys.argv[1].split(".")[0]
imgsz = (640, 640)
model = YOLO(f"{name}.pt")

model.export(format="mlmodel", imgsz=imgsz)
os.system(f"xcrun coremlcompiler compile {name}.mlmodel ./")
