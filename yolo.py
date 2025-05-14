import os
import cv2

from ultralytics import YOLO

ground_truth = {}
correct = 0
count = 0

def get_image_files_from_directory(directory):
    """Returns a list of image file paths from the given directory."""
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(supported_extensions)]

def read_ground_truth(ground_truth_file):
    """Reads the ground truth from the given file."""
    global ground_truth

    with open("fileAMKweight.csv", "r") as file:
        for line in file:
            filename, id_val, weight = line.strip().split(",")
            ground_truth[filename] = {
                "id": id_val.strip(),
                "weight": float(weight),
                # You can add more keys like "groundtruth" if needed
            }


def main():
    global count
    global correct

    model = YOLO('best_label_and_scale_display.pt')
    #model_weight = YOLO('best_created_data.pt')
    model_weight = YOLO('best_weights_full1.pt')
    #model_weight = YOLO("/home/lukas/ausland/course/digital-signal-processing/modelTraining/runs/detect/train7/weights/best.pt")

    # Load original image
    original_img = cv2.imread("dataset/20250403_114728.jpg")
    original_img = cv2.rotate(original_img, cv2.ROTATE_180)

    # Run inference
    results = model(original_img)

    # Get class names from the model
    class_names = model.names  # e.g., {0: 'label', 1: 'scale_display'}


    amks = []
    weights = []
    barcodes = []

    global ground_truth
    read_ground_truth(ground_truth)
    for i, result in enumerate(results):
        boxes = result.boxes
        for j, box in enumerate(boxes):
            # Get bounding box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Get class ID and class name
            class_id = int(box.cls[0].cpu().numpy())
            class_name = class_names[class_id]

            # Crop detected region
            cropped = original_img[y1:y2, x1:x2]

            # Save with class in filename
            filename = f"cropped_{class_name}_{i}_{j}.jpg"
            #cv2.imwrite(filename, cropped)
            #print(f"Saved {filename} for class '{class_name}'")

            #cv2.imshow(f"YOLO result for class '{class_name}'", cropped)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            if class_name == 'scale_display':
                # results = model('/home/lukas/abroad/courses/digitalSignalProcessing/digital-signal-processing/dataset/20250403_114728.jpg')  # predict on an image

                results_weight = model_weight(cropped)

                # Get class names from the model
                class_names_weights = model_weight.names  # e.g., {0: 'label', 1: 'scale_display'}


                for digit in results_weight:
                    boxes_weight = digit.boxes
                    for j, box_weight in enumerate(boxes_weight):
                        # Get class ID and class name
                        class_id = int(box_weight.cls[0].cpu().numpy())
                        class_name = class_names_weights[class_id]
                        print("Found weights: ", class_name)
                    #digit.show()


    ###  HERE LOOP FOR ALL IMAGES
    dataset_dir = "dataset"
    image_files = get_image_files_from_directory(dataset_dir)

    if not image_files:
        print("No images found in the dataset directory.")
        return

    for image_path  in image_files:
        print("Processing image: ", image_path)

        original_img = cv2.imread(image_path)
        original_img = cv2.rotate(original_img, cv2.ROTATE_180)

        # Run inference
        results = model(original_img)

        # Get class names from the model
        class_names = model.names  # e.g., {0: 'label', 1: 'scale_display'}

        for i, result in enumerate(results):
            boxes = result.boxes
            for j, box in enumerate(boxes):
                # Get bounding box
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Get class ID and class name
                class_id = int(box.cls[0].cpu().numpy())
                class_name = class_names[class_id]

                # Crop detected region
                cropped = original_img[y1:y2, x1:x2]

                # Save with class in filename
                filename = f"cropped_{class_name}_{i}_{j}.jpg"
                #cv2.imwrite(filename, cropped)
                #print(f"Saved {filename} for class '{class_name}'")

                #cv2.imshow(f"YOLO result for class '{class_name}'", cropped)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                if class_name == 'scale_display':
                    results_weight = model_weight(cropped)
                    class_names_weights = model_weight.names  # {0: '1', 1: '2-dot', 2: '3-kg', ...}

                    best_dot = (" Nan ", 0.0)  # (class_name, confidence)
                    best_number = (" Nan ", 0.0)
                    best_kg = (" Nan ", 0.0)

                    for digit in results_weight:
                        boxes_weight = digit.boxes
                        for box_weight in boxes_weight:
                            class_id = int(box_weight.cls[0].cpu().numpy())
                            class_name = class_names_weights[class_id]
                            conf = float(box_weight.conf[0].cpu().numpy())

                            if "-dot" in class_name:
                                sub = class_name.replace("-dot", "")
                                if conf > best_dot[1]:
                                    best_dot = (sub, conf)

                            elif "-kg" in class_name:
                                sub = class_name.replace("-kg", "")
                                if conf > best_kg[1]:
                                    best_kg = (sub, conf)

                            else:
                                if conf > best_number[1]:
                                    best_number = (class_name, conf)

                            print("Found class:", class_name, "with confidence:", conf)

                    first = best_dot[0]
                    second = best_number[0]
                    third = best_kg[0]

                    print("Selected:", first, second, third)

                    print("Found weight: " + first + "." + second + "" + third + " kg")
                    print("Ground truth: ", ground_truth[image_path]["weight"])
                    if first != " Nan " and second != " Nan " and third != " Nan ":
                        tmp_weight = float(first + "." + second + "" + third)
                        if tmp_weight == ground_truth[image_path]["weight"]:
                            correct += 1
                    count += 1

    print("Accuracy: ", correct / count)
    print("Correct =  ", correct)
    print("Counter = ", count)




if __name__ == "__main__":
    main()

