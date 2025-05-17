import cv2
import easyocr
import time
import os
import re
from pyzbar import pyzbar
import template_matching
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


results = []


DEBUG_BARCODE = False
DEBUG_BARCODE_FOLDER = "debug_results_barcode"
RASPI = False


found_amk = 0


def initialize_ocr(languages=None, gpu=False):
    """
    Initializes an OCR reader with specified languages and GPU acceleration support.

    This function creates an EasyOCR reader instance configured with the desired
    languages for text recognition and an option to enable GPU-based processing.

    Parameters:
    languages : list[str], optional
        List of languages for OCR. Defaults to ['en'] if not provided.
    gpu : bool
        Whether to enable GPU acceleration. Defaults to False.

    Returns:
    easyocr.Reader
        The initialized OCR reader configured with the specified languages and GPU
        setting.
    """
    if languages is None:
        languages = ['en']
    return easyocr.Reader(languages, gpu=gpu)

def find_amk_codes(text_lines):
    """
    Extracts specific AMK codes from a list of text lines using a regex pattern.

    The function searches through a given list of text lines to find all matches of
    AMK codes. AMK codes adhere to a specific pattern consisting of the prefix
    "AMK" (case-insensitive) followed by an optional space or hyphen, and a series
    of 5 digits. The function returns all matching codes found across the input
    lines.

    Parameters:
        text_lines (list of str): The list of text lines to search for AMK codes.

    Returns:
        list of str: A list of matched AMK codes extracted from the input text
        lines.
    """
    #pattern = r'AMK[-\s]?\d+'

    #pattern = r'\b[Aa]?[Mm][Kk][- ]?\d{5}\b'
    #pattern = r'\b([A-Z]?[Mm]?[Kk]?)+\d{4,6}\b'
    #pretty good
    #pattern = r'\b(?=[A-Z]*[AMK])[A-Z]{1,4}[- ]?\d{5}\b'
    pattern = r'[^A-Z0-9]?([AMK]{1,3}[- ]?\d{5})[^A-Z0-9]?'

    matches = []

    for line in text_lines:
        found = re.findall(pattern, line, flags=re.IGNORECASE)
        matches.extend(found)

    return matches


def perform_ocr(reader, image_path, roi=None, top_left=None, bottom_right=None):
    """
    Performs Optical Character Recognition (OCR) on a specified image using a given
    OCR reader. It optionally allows defining a region of interest (ROI) either through
    explicit ROI values or rectangular bounding box coordinates.

    Parameters:
    reader : Any
        An OCR reader object used to process the image and extract text.
    image_path : str
        The path to the image file on which OCR will be performed.
    roi : Optional[Tuple[int, int, int, int]]
        A tuple specifying the region of interest in the image as (x, y, width, height).
        This parameter is optional and is used if top_left and bottom_right are not provided.
    top_left : Optional[Tuple[int, int]]
        Coordinates of the top-left corner of a rectangular region of interest. Optional.
    bottom_right : Optional[Tuple[int, int]]
        Coordinates of the bottom-right corner of a rectangular region of interest. Optional.

    Returns:
    Tuple[List[str], Optional[numpy.ndarray]]
        A tuple where the first element is a list of extracted text strings and the second element
        is the modified image with bounding boxes and text annotations. If the image cannot be read,
        returns (None, None).

    Raises:
    FileNotFoundError
        If the specified image file is not found or cannot be read.

    Notes:
    If both `roi` and `top_left`/`bottom_right` parameters are provided, only the `top_left`
    and `bottom_right` parameters are considered for defining the region of interest.

    The function preprocesses the input image by rotating it 180 degrees and scaling the
    specified region of interest. The OCR reader then analyzes the preprocessed region for text
    detection and extraction. Bounding boxes and recognized text are drawn on the region of
    interest in the output image.

    The time taken to perform the OCR operation is printed to the standard output, along with
    the name of the input image file being processed.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    image = cv2.rotate(image, cv2.ROTATE_180)


    # Crop the region of interest if specified
    if roi is not None and top_left is None and bottom_right is None:
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w].copy()
    else:
        roi_image = image.copy()

    if top_left is not None and bottom_right is not None:
        image = cv2.imread(image_path)
        image = cv2.rotate(image, cv2.ROTATE_180)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        roi_image = gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    scale_factor = 2.0
    roi_image = cv2.resize(roi_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    #preprocssed = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    #preprocssed = preprocess_image(roi_image)
    preprocssed = roi_image
    start_time = time.time()
    results = reader.readtext(preprocssed)
    end_time = time.time()

    print(f"OCR on {os.path.basename(image_path)} completed in {end_time - start_time:.2f} seconds")

    # Draw bounding boxes and text on roi_image
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(roi_image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(roi_image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    extracted_texts = [text for (_, text, _) in results]
    return extracted_texts, roi_image


def save_ocr_results(image_path, texts, annotated_image, output_dir="results"):
    """
    Saves OCR results including detected text, "amk" code matches, and annotated
    images to an output directory. Organizes saved data into text and picture files
    within the specified directory. Ensures directory structure is created if it
    doesn't exist. Returns the "amk" code matches found in the provided text.

    Args:
        image_path (str): File path to the input image being processed.
        texts (list[str]): List of extracted text lines from the OCR process.
        annotated_image (numpy.ndarray): Image with OCR annotations added.
        output_dir (str): Directory where the results will be saved. Defaults to "results".

    Returns:
        list[str]: List of extracted "amk" code matches found in the provided text.
    """
    os.makedirs(output_dir, exist_ok=True)
    global found_amk
    pictures_dir = os.path.join(output_dir, "pictures")
    os.makedirs(pictures_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    base_name, ext = os.path.splitext(image_name)

    # Extract amk codes from text
    amk_matches = find_amk_codes(texts)

    # Save OCR text and matches
    text_output_path = os.path.join(output_dir, base_name + ".txt")
    with open(text_output_path, 'w', encoding='utf-8') as f:
        for line in texts:
            f.write(line + "\n")
        if amk_matches:
            f.write("\n--- amk Matches ---\n")
            print("--- amk Matches ---")
            found_amk += 1
            for match in amk_matches:
                f.write(match + "\n")
                print(match)

    print(f"Saved OCR text to: {text_output_path}")

    # Save annotated image
    image_output_path = os.path.join(pictures_dir, base_name + ".jpg")
    cv2.imwrite(image_output_path, annotated_image)
    print(f"Saved annotated image to: {image_output_path}")
    return amk_matches



def get_image_files_from_directory(directory):
    """
    Gets the list of image files from a specified directory. Only files with supported
    extensions are considered. The function ensures that the returned list of file
    paths is sorted alphabetically.

    Parameters:
        directory (str): Path to the directory from which image files are to be
        retrieved.

    Returns:
        list[str]: A list of file paths pointing to the image files in the
        specified directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is not a directory.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(supported_extensions)]

def perform_ocr_with_rotation(reader, image, roi=None):
    """
    Performs OCR on an image at multiple rotations and optionally within a specified region of interest (ROI),
    returns the recognized text alongside a visualization of the processed image.

    Parameters:
    reader : Any
        An OCR reader object capable of performing text recognition.
    image : numpy.ndarray
        The input image to process for OCR.
    roi : tuple[int, int, int, int] | None, optional
        A tuple representing the region of interest as (x, y, width, height), or None if the entire image is used.

    Returns:
    tuple[list[str] | None, numpy.ndarray | None]
        A tuple where the first element is a list of recognized texts if successful, or None if no matches are found,
        and the second element is the processed image with annotations, or None if no matches are found.
    """
    rotations = {
        0: image,
        180: cv2.rotate(image, cv2.ROTATE_180),
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }

    for angle, rotated_img in rotations.items():
        if roi:
            x, y, w, h = roi
            roi_image = rotated_img[y:y+h, x:x+w].copy()
        else:
            roi_image = rotated_img.copy()

        scale_factor = 2.0
        roi_image = cv2.resize(roi_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # preprocssed = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        # preprocssed = preprocess_image(roi_image)
        preprocssed = roi_image
        start_time = time.time()
        results = reader.readtext(preprocssed)
        end_time = time.time()

        extracted_texts = [text for (_, text, _) in results]
        amk_matches = find_amk_codes(extracted_texts)

        print(f"Rotation {angle}° - Found {len(amk_matches)} amk matches - OCR time: {end_time - start_time:.2f}s")

        if amk_matches:
            # Draw bounding boxes and text
            for (bbox, text, prob) in results:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(roi_image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(roi_image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return extracted_texts, roi_image

    print("No amk code found in any rotation.")
    return None, None

def rotate_image(image, angle):
    """
    Rotates an image by a specified angle around its center.

    This function takes an image and rotates it by the given angle, while keeping
    the center of the image fixed. The output image retains the same dimensions
    as the input image. The resulting rotation does not involve cropping, and
    preserves the full image content within the original dimensions.

    Parameters:
        image (numpy.ndarray): The input image to be rotated. It is expected to
            be a multidimensional array where the first two dimensions represent
            the height and width of the image, respectively.
        angle (float): The angle, in degrees, by which the image will be
            rotated. Positive values indicate counter-clockwise rotation,
            while negative values indicate clockwise rotation.

    Returns:
        numpy.ndarray: The rotated image as the same type and size as the input
            image, with its orientation changed according to the specified angle.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))

    return rotated


def decode_barcodes(image):
    """
    Decodes barcodes from an image and returns their data, type, and position.

    This function takes an image as input, decodes the barcodes within it, and
    returns a list of tuples. Each tuple contains the decoded barcode data, the
    type of barcode, and its rectangular position within the image.

    Args:
        image: An image containing one or more barcodes.

    Returns:
        List of tuples where each tuple consists of:
        - barcode data as a string
        - barcode type as a string
        - rectangular coordinates of the barcode as a tuple (x, y, w, h)
    """
    barcodes = pyzbar.decode(image)
    results = []
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        rect = barcode.rect  # x, y, w, h
        results.append((barcode_data, barcode_type, rect))
    return results


def draw_barcodes(image, barcodes):
    """
    Draws bounding boxes and data on an image for detected barcodes.

    This function processes an input image and overlays bounding boxes and
    text information for each detected barcode. The information includes
    the barcode's value and its type. The bounding boxes are drawn using a
    green rectangle while the text is displayed above the bounding box.

    Parameters:
    image: numpy.ndarray
        The input image where barcodes are to be visualized.
    barcodes: list of tuple
        A list of barcode information where each tuple contains:
        - data (str): The value or content of the barcode.
        - btype (str): The type of the barcode (e.g., QR Code, EAN-13).
        - rect (tuple of int): The bounding rectangle (x, y, width, height)
          indicating the position of the barcode in the image.

    Returns:
    numpy.ndarray
        The resulting image with barcodes highlighted and labeled.

    """
    for data, btype, rect in barcodes:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'{btype}: {data}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image


def process_image_barcode(image):
    """
    Processes an image to detect and decode barcodes at various rotations.

    This function attempts to identify barcodes within the provided image by iteratively
    rotating it to four preset angles (0°, 180°, 90°, 270°) and decoding for barcodes
    at each rotation. If a barcode is found, it prints the rotation angle at which it
    was detected and optionally saves a debug image illustrating the detected barcode.

    Attributes:
        DEBUG_BARCODE (bool): A global flag that enables or disables saving
            debug images showcasing detected barcodes.
        DEBUG_BARCODE_FOLDER (str): A global folder path where debug images
            are saved if DEBUG_BARCODE is enabled.

    Parameters:
        image: The input image to process for barcode detection.

    Returns:
        A list of detected barcodes. Each barcode is represented by its associated
        data decoded from the image. Returns an empty list if no barcodes are found.
    """
    for angle in [0, 180, 90, 270]:
        rotated = rotate_image(image, angle)
        barcodes = decode_barcodes(rotated)
        if barcodes:
            print(f"[✓] Found Barcode in (rotation: {angle}°)")

            if DEBUG_BARCODE:
                debug_image = draw_barcodes(rotated.copy(), barcodes)
                if not os.path.exists(DEBUG_BARCODE_FOLDER):
                    os.makedirs(DEBUG_BARCODE_FOLDER)
                timestr = time.strftime("%Y%m%d-%H%M%S")
                debug_filename = os.path.join(DEBUG_BARCODE_FOLDER, f"debug_{angle}_{os.path.basename(timestr)}.png")
                cv2.imwrite(debug_filename, debug_image)
                print(f"    Saved debug image: {debug_filename}")

            return barcodes

    print(f"[x] No barcode found!")
    return []


def analyze_image(dataset_path):
    """
    Analyzes images within a specified dataset directory by identifying regions of interest,
    performing object detection using pre-trained YOLO models, extracting relevant data such
    as scale weights, barcodes, and AMKs, then saves the results into a CSV file.

    Attributes:
        roi: tuple of int
            Region of interest dimensions in the format (x, y, width, height) used for
            processing the images.
        scale_roi: tuple of int
            Specific region of interest for detecting scales in the images.
        amks: list of str
            Contains extracted AMK-related information from images.
        weights: list of str
            Contains weight information detected from scales in the images.
        barcodes: list of str
            Contains barcode data identified during image analysis.

    Parameters:
        dataset_path: str
            The path to the directory containing image files to analyze.

    Returns:
        str
            The path to the location of the generated CSV file containing analyzed results.

    Raises:
        This function does not explicitly raise errors but depends on external library
        behavior to handle any exceptions during file I/O, image processing, or object
        detection tasks.
    """
    dataset_dir = dataset_path
    #results_dir = "results"
    scale_template_path = "scale_display.png"
    image_files = get_image_files_from_directory(dataset_dir)

    if not image_files:
        print("No images found in the dataset directory.")
        return



    # Define the region of interest: (x, y, width, height)
    roi = (500, 0, 2500, 2500)
    scale_roi = (1500, 350, 1000, 500)

    amks = []
    weights = []
    barcodes = []

    # Load a model
    base_dir = Path('models')
    model = YOLO(base_dir / 'best_label_and_scale_display.pt')
    model_weight = YOLO(base_dir / 'best_weights_full1.pt')
    class_names = model.names

    amks.append("AMKs")
    weights.append("Weights")
    barcodes.append("Barcodes")
    for image_path in image_files:
        print(f"===========================================================================================================")
        print(f"[*] Processing image: {image_path}")
        original_img = cv2.imread(image_path)
        original_img = cv2.rotate(original_img, cv2.ROTATE_180)

        results_model = model(original_img)
        weight = "Nan"
        for i, result in enumerate(results_model):
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
                # cv2.imwrite(filename, cropped)
                # print(f"Saved {filename} for class '{class_name}'")

                # cv2.imshow(f"YOLO result for class '{class_name}'", cropped)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
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
                    if first != " Nan " and second != " Nan " and third != " Nan ":
                        found_weight = float(first + "." + second + "" + third)

                    if first == " Nan " or second == " Nan " or third == " Nan ":
                        print('Could not find a valid weight!')
                        weight = "Nan"
                    else:
                        weight = float(f"{first}.{second}{third}")


                if class_name == 'label':

                    # Barcode
                    found_barcode = process_image_barcode(cropped)
                    if found_barcode:
                        for data, btype, rect in found_barcode:
                            print(f"{image_path}: [{btype}] {data}")
                    else:
                        print(f"{image_path}: No barcode found.")

                    amk = None
                    if not RASPI:
                        # AMK
                        reader = initialize_ocr(['en', 'ko'], True)
                        texts, annotated_image = perform_ocr_with_rotation(reader, cropped)

                        if texts is not None and annotated_image is not None:
                            amk = save_ocr_results(image_path, texts, annotated_image, "results")

        barcode_data = "; ".join(
            [f"[{btype}] {data}" for data, btype, rect in found_barcode]) if found_barcode else "None"

        results.append({
            "image_path": image_path,
            "barcode": barcode_data,
            "weight": weight,
            "amk": amk if amk else "None"
        })

    df = pd.DataFrame(results)
    results_path = Path(dataset_path) / 'results.csv'
    df.to_csv(results_path, index=False)

    return str(results_path)



