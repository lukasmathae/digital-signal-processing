from ultralytics import YOLO

# Load a model
model = YOLO('best_label_and_scale_display.pt')  # load a pretrained model (recommended for training)

# Use the model

#results = model('/home/lukas/abroad/courses/digitalSignalProcessing/digital-signal-processing/dataset/20250403_114728.jpg')  # predict on an image

results = model(["dataset/20250403_114728.jpg",
                 "dataset/20250403_114804.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

# Load a model
model = YOLO('best_weights.pt')  # load a pretrained model (recommended for training)

# Use the model

#results = model('/home/lukas/abroad/courses/digitalSignalProcessing/digital-signal-processing/dataset/20250403_114728.jpg')  # predict on an image

results = model(["dataset_display/20250403_114728.jpg",
                 "dataset_display/20250403_114804.jpg"])
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk