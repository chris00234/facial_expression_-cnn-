from ultralytics import YOLO
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import os


# model = YOLO("yolov8s.pt")
# model.train(data = "dataset.yaml", epochs= 3)

model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')
# # Load a model
model = YOLO(model_path)  # load a custom model

results = model.predict(source = 0, show = True)