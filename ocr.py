import cv2
import easyocr
import time


def initialize_camera(width=640, height=480):
    """Initializes the camera and sets the resolution."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Error: Could not open camera.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def initialize_ocr(languages=None):
    """Initializes the EasyOCR reader with given languages."""
    if languages is None:
        languages = ['en', 'ko']
    return easyocr.Reader(languages)


def perform_ocr(reader, frame):
    """Runs OCR on the given frame and returns results and processing time."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    results = reader.readtext(gray)
    end_time = time.time()
    print(f"OCR Results: {results}")
    print(f"OCR Time: {end_time - start_time:.2f} seconds")
    return results


def main():
    reader = initialize_ocr(['en', 'ko'])
    camera = initialize_camera()

    print("Press SPACE to capture and run OCR, press 'q' to quit.")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            perform_ocr(reader, frame)

        cv2.imshow("Live Camera Feed", frame)

        if key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
