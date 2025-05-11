import os
import cv2

from ultralytics import YOLO

ground_truth = {}

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
    model = YOLO('best_label_and_scale_display.pt')
    model_weight = YOLO('best_weights_mmodel.pt')

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
                    digit.show()


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

                    # Get class names from the model
                    class_names_weights = model_weight.names  # e.g., {0: 'label', 1: 'scale_display'}


                    first = " Nan "
                    second = " Nan "
                    third = " Nan "
                    for digit in results_weight:
                        boxes_weight = digit.boxes
                        for j, box_weight in enumerate(boxes_weight):
                            # Get class ID and class name
                            class_id = int(box_weight.cls[0].cpu().numpy())
                            class_name = class_names_weights[class_id]
                            if "-dot" in  class_name and first == " Nan ":
                                sub = class_name.replace("-dot", "")
                                first = sub
                            elif "-kg" in class_name and third == " Nan ":
                                sub = class_name.replace("-kg", "")
                                third = sub
                            elif "-dot" not in class_name and "-kg" not in class_name and second == " Nan ":
                                second = class_name
                            if first != " Nan " and second != " Nan " and third != " Nan ":
                                break

                            print("Found classes: ", class_name)

                    print("Found weight: " + first + "." + second + "" + third + " kg")
                    print("Ground truth: ", ground_truth[image_path]["weight"])
                    #tmp_weight = float(first + "." + second + "" + third)




if __name__ == "__main__":
    main()

