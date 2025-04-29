# TODO try to find the scale and then do the comparision
import cv2
import os

#TEMPLATE_DIR = './scale_digit_templates/'
templates = {}

validation = \
[0.28,
0.46,
0.46,
0.74,
0.58,
0.78,
0.22,
0.44,
0.18,
2.66,
1.06,
4.72,
0.66,
1.18,
1.20,
1.28,
0.92,
2.50,
0.72,
0.42,
0.62,
0.48]

validation_counter = 0


def load_templates(digits_template_dir):
    for entry in os.scandir(digits_template_dir):
        if entry.is_file():
            digit = os.path.splitext(entry.name)
            if digit[0] != "3":
                template = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)
                templates[str(digit[0])] = template




def find_display_roi(gray_img):
    """Automatically detect ROI containing 7-segment digits."""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 20 < w < 150 and 50 < h < 200 and 0.2 < aspect_ratio < 1.2:
            digit_regions.append((x, y, w, h))

    if not digit_regions:
        print("⚠️ No digit-like regions found.")
        return None

    x_vals = [x for x, _, _, _ in digit_regions]
    y_vals = [y for _, y, _, _ in digit_regions]
    x_ends = [x + w for x, _, w, _ in digit_regions]
    y_ends = [y + h for _, y, _, h in digit_regions]

    x_min = min(x_vals)
    y_min = min(y_vals)
    x_max = max(x_ends)
    y_max = max(y_ends)

    return (x_min - 10, y_min - 10, x_max - x_min + 20, y_max - y_min + 20)




def get_image_files_from_directory(directory):
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(supported_extensions)]

def find_scale_display(full_image_path, template_image_path, roi, debug=False):
    # Load the full image and the template
    x, y, w, h = roi
    full_image_gray = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    full_image_color_rotated = cv2.rotate(full_image_gray, cv2.ROTATE_180)

    roi_image = full_image_color_rotated[y:y + h, x:x + w].copy()
    template_gray = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    # Perform template matching
    res = cv2.matchTemplate(roi_image, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # Define the bounding box of the matched region
    h, w = template_gray.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw rectangle for visualization
    cv2.rectangle(roi_image, top_left, bottom_right, (0, 255, 0), 2)

    # Extract and show/save ROI
    display_region = roi_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Optional: save the debug result
    #cv2.imwrite(debug_output_path, full_image_color)
    #cv2.imshow("Matched Template Region", full_image_color)
    if debug:
        cv2.imshow("Extracted ROI ", display_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return display_region, top_left, bottom_right, max_val


def compare_diplay_to_templates(scale_display_img_gray, matching_debug):
    # Define a threshold for matching (you can tweak this value)
    MATCH_THRESHOLD = 0.8  # for TM_CCOEFF_NORMED, values closer to 1 are better matches
    matched_digits = {}
    for digit, template in templates.items():
        try:
            if matching_debug:
                cv2.imshow("Gray Image Scale ", scale_display_img_gray)
                cv2.imshow("Template", template)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            '''
            methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                       'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            '''

            # Perform template matching
            res = cv2.matchTemplate(scale_display_img_gray, template, cv2.TM_CCOEFF)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # Check if the match is good enough
            if max_val < MATCH_THRESHOLD:
                if matching_debug: print(f"Skipping {digit}, poor match (score: {max_val:.2f})")
                continue  # Skip to the next template

            if matching_debug: print(f"Found good match for {digit} (score: {max_val:.2f})")
            matched_digits[digit] = max_val

            # Define the bounding box of the matched region
            h, w = template.shape[:2]
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Draw rectangle for visualization
            cv2.rectangle(scale_display_img_gray, top_left, bottom_right, (0, 255, 0), 2)

            # Extract and show/save ROI
            match_numbers = scale_display_img_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            if matching_debug:
                cv2.imshow("Matching", match_numbers)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing {digit}: {e}")
            continue

    matched_digits = {key: val for key, val in sorted(matched_digits.items(), key=lambda ele: ele[1], reverse=True)}
    counter = 0
    res = {}
    first = " Nan "
    second = " Nan "
    third = " Nan "
    for k, v in list(matched_digits.items()):
        if "-dot" in k and first == " Nan ":
            sub = k.replace("-dot", "")
            first = sub
        elif "-kg" in k and third == " Nan ":
            sub = k.replace("-kg", "")
            third = sub
        elif "-dot" not in k and "-kg" not in k and second == " Nan ":
            second = k
        if first != " Nan " and second != " Nan " and third != " Nan ":
            break

    return matched_digits, first, second, third


def template_matching(image, scale_template_path, digits_template_dir, roi, debug_scale_display=False, debug_template_matching=False):
    #dataset_dir = "/home/lukas/abroad/courses/digitalSignalProcessing/digital-signal-processing/dataset"
    #scale_template_path = "scale_display.png"
    #image_files = get_image_files_from_directory(dataset_dir)

    if not image:
        print("No images found in the dataset directory.")
        return

    load_templates(digits_template_dir)

    # Define the region of interest: (x, y, width, height)
    #roi = (500, 0, 2500, 2500)
    #roi = (1500, 350, 1000, 500)
    '''
    for image_path in image_files:
        display_region, top_left, bottom_right, max_val = find_scale_display(image_path, scale_template_path, roi, debug_scale_display)
        matched_digits, first, second, third = compare_diplay_to_templates(display_region, debug_template_matching)
        print(f"==================================================")
        #global validation_counter
        print(f"Found for {image_path} following digits: {matched_digits}")
        #print(
        #    f"Weight: {first} . {second} {third} kg (should be counter({validation_counter}): {validation[validation_counter]})")
        #validation_counter += 1
        print(
            f"Weight: {first} . {second} {third} kg)")
    '''

    display_region, top_left, bottom_right, max_val = find_scale_display(image, scale_template_path, roi,
                                                                         debug_scale_display)
    matched_digits, first, second, third = compare_diplay_to_templates(display_region, debug_template_matching)
    # global validation_counter
    print(f"Found for following digits: {matched_digits}")
    # print(
    #    f"Weight: {first} . {second} {third} kg (should be counter({validation_counter}): {validation[validation_counter]})")
    # validation_counter += 1
    print(
        f"Weight: {first} . {second} {third} kg")

    cv2.destroyAllWindows()