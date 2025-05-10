import os
import cv2

from ultralytics import YOLO


def get_image_files_from_directory(directory):
    """Returns a list of image file paths from the given directory."""
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(supported_extensions)]



def main():

    dataset_dir = "dataset"
    results_dir = "results"
    scale_template_path = "scale_display.png"
    image_files = get_image_files_from_directory(dataset_dir)

    if not image_files:
        print("No images found in the dataset directory.")
        return

    '''
    # Load a model
    model = YOLO('best_label_and_scale_display.pt')  # load a pretrained model (recommended for training)

    # Use the model

    #results = model('/home/lukas/abroad/courses/digitalSignalProcessing/digital-signal-processing/dataset/20250403_114728.jpg')  # predict on an image

    results = model(["dataset/20250403_114728.jpg"])
    for result in results:
        # Process results list
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        result.save(filename="result_labelDisplay.jpg")  # save to disk
        print(result.masks)  # print the Masks object containing the detected instance masks

        # You can access the original image with drawings using `result.plot()`
        image_with_boxes = result.plot()  # Returns a numpy array (BGR)

        # Save with OpenCV (if needed)
        cv2.imwrite("result_with_boxes.jpg", image_with_boxes)

        # Now you can use OpenCV to do further processing on `image_with_boxes`
        # For example, convert to grayscale:
        gray = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2GRAY)

        # Or show it with OpenCV
        cv2.imshow("YOLO result", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    # Load a model
    model = YOLO('best_weights.pt')  # load a pretrained model (recommended for training)

    # Use the model

    #results = model('/home/lukas/abroad/courses/digitalSignalProcessing/digital-signal-processing/dataset/20250403_114728.jpg')  # predict on an image

    results = model(["dataset_display/20250403_114728.jpg"])
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        result.save(filename="result_weight.jpg")
        print(result.masks)  # print the Masks object containing the detected instance masks
'''
    # Load the model
    model = YOLO('best_label_and_scale_display.pt')

    # Load original image
    original_img = cv2.imread("dataset/20250403_114728.jpg")
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
                # Load a model
                model_weight = YOLO('best_weights.pt')  # load a pretrained model (recommended for training)

                # Use the model

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



if __name__ == "__main__":
    main()

