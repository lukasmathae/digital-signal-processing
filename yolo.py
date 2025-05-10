from ultralytics import YOLO

# Load a model
model = YOLO('/content/runs/detect/train2/weights/best.pt')  # load a pretrained model (recommended for training)

# Use the model

results = model('/home/lukas/abroad/courses/digitalSignalProcessing/digital-signal-processing/dataset/20250403_114728.jpg', )  # predict on an image
