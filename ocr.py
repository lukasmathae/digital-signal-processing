import cv2
import easyocr
import time
import os
import re

found_amk = 0


def initialize_ocr(languages=None, gpu=False):
    """Initializes the EasyOCR reader with given languages."""
    if languages is None:
        languages = ['en']
    return easyocr.Reader(languages, gpu=gpu)

def find_amk_codes(text_lines):
    """
    Finds patterns similar to AMK+5 digits in OCR text lines.
    Accepts: AMK12345, AMK-12345, amk 12345, mk12345, etc.
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

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def perform_ocr(reader, image_path, roi=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    image = cv2.rotate(image, cv2.ROTATE_180)

    # Crop the region of interest if specified
    if roi is not None:
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w].copy()
    else:
        roi_image = image.copy()

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
    """Saves OCR text and annotated image to the results directory."""
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
            print(f"=================================(Found {len(amk_matches)} amk+number patterns)=======================")
            f.write("\n--- amk Matches ---\n")
            found_amk += 1
            for match in amk_matches:
                f.write(match + "\n")
                print(match)

    print(f"Saved OCR text to: {text_output_path}")

    # Save annotated image
    image_output_path = os.path.join(pictures_dir, base_name + ".jpg")
    cv2.imwrite(image_output_path, annotated_image)
    print(f"Saved annotated image to: {image_output_path}")



def get_image_files_from_directory(directory):
    """Returns a list of image file paths from the given directory."""
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(supported_extensions)]

def perform_ocr_with_rotation(reader, image_path, roi=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    rotations = {
        180: cv2.rotate(image, cv2.ROTATE_180),
        0: image,
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



def main():
    dataset_dir = "dataset"
    results_dir = "results"
    image_files = get_image_files_from_directory(dataset_dir)

    if not image_files:
        print("No images found in the dataset directory.")
        return

    reader = initialize_ocr(['en', 'ko'], True)

    # Define the region of interest: (x, y, width, height)
    roi = (500, 0, 2500, 2500)

    for image_path in image_files:
        texts, annotated_image = perform_ocr(reader, image_path, roi)
        #texts, annotated_image = perform_ocr_with_rotation(reader, image_path, roi)

        if texts is not None and annotated_image is not None:
            save_ocr_results(image_path, texts, annotated_image, results_dir)

    print("Done. Found " + str(found_amk) + " amk+number patterns.")


if __name__ == "__main__":
    main()
