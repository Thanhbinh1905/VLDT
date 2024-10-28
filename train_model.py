
from ultralytics import YOLO
import torch
torch.cuda.empty_cache()


if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data="D:/WorkSpace/PycharmProjects/pythonProject/data.yaml", epochs=50, imgsz=640, device='cuda', workers=4, batch = 32)
