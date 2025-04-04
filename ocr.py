import cv2
import easyocr
import time

# Initialize EasyOCR reader (optimize for English only to improve speed)
reader = easyocr.Reader(['en', 'ko'])

# Open the camera (0 for default camera)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set lower resolution to improve performance on Raspberry Pi
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect when user is pressing 'space' key
    if cv2.waitKey(1) & 0xFF == ord(' '):


        # Convert frame to grayscale (EasyOCR performs better on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run OCR on the frame
        # start_time = time.time()
        # results = reader.readtext(gray)
        # print(results)
        # end_time = time.time()
            # Run OCR on the frame
        start_time = time.time()
        results = reader.readtext(gray)
        end_time = time.time()

        print(f"OCR Results: {results}")  # 👈 Print detected results for debugging

        if not results:
            cv2.putText(frame, "No text detected", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # Display results on the frame
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            
            # Draw bounding box
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            
            # Display text
            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show FPS
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Camera Only", frame)
    # cv2.imshow("OCR Camera", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
