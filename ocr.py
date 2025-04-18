import cv2
import easyocr
import time
import os


def initialize_ocr(languages=None):
    """Initializes the EasyOCR reader with given languages."""
    if languages is None:
        languages = ['en', 'ko']
    return easyocr.Reader(languages)


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

    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    results = reader.readtext(gray)
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
    pictures_dir = os.path.join(output_dir, "pictures")
    os.makedirs(pictures_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    base_name, ext = os.path.splitext(image_name)

    # Save OCR text
    text_output_path = os.path.join(output_dir, base_name + ".txt")
    with open(text_output_path, 'w', encoding='utf-8') as f:
        for line in texts:
            f.write(line + "\n")
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


def main():
    dataset_dir = "dataset"
    results_dir = "results"
    image_files = get_image_files_from_directory(dataset_dir)

    if not image_files:
        print("No images found in the dataset directory.")
        return

    reader = initialize_ocr(['en', 'ko'])

    # Define the region of interest: (x, y, width, height)
    roi = (500, 0, 2500, 2500)

    for image_path in image_files:
        texts, annotated_image = perform_ocr(reader, image_path, roi)
        if texts is not None and annotated_image is not None:
            save_ocr_results(image_path, texts, annotated_image, results_dir)


if __name__ == "__main__":
    main()
