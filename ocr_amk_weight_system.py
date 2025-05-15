import cv2
import pytesseract
import pandas as pd
import os
import re

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def extract_text(image_path):
    processed_img = preprocess_image(image_path)
    return pytesseract.image_to_string(processed_img)

def extract_amk_and_weight(text):
    amk_match = re.search(r'AMK\s?-?\s?(\d+)', text, re.IGNORECASE)
    amk = f"AMK{amk_match.group(1)}" if amk_match else "N/A"

    weight_match = re.search(r'(\d+\.\d+)\s?kg', text, re.IGNORECASE)
    if weight_match:
        weight = round(float(weight_match.group(1)) * 1000)
    else:
        weight = "N/A"

    return amk, weight

def process_dataset(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            text = extract_text(image_path)
            amk, weight = extract_amk_and_weight(text)
            data.append({
                'filename': filename,
                'amk_predicted': amk,
                'weight_predicted': weight
            })

    df = pd.DataFrame(data)
    df.to_csv('predicted.csv', index=False)
    print("✅ predicted.csv created")

def compare_with_ground_truth(predicted_csv, ground_truth_csv):
    pred_df = pd.read_csv(predicted_csv)
    gt_df = pd.read_csv(ground_truth_csv)

    merged = pd.merge(pred_df, gt_df, on='filename', suffixes=('_predicted', '_truth'))
    merged['amk_match'] = merged['amk_predicted'] == merged['amk_truth']
    merged['weight_match'] = merged['weight_predicted'] == merged['weight_truth']

    print(merged[['filename', 'amk_predicted', 'amk_truth', 'amk_match',
                  'weight_predicted', 'weight_truth', 'weight_match']])

    merged.to_csv('comparison_result.csv', index=False)
    print("✅ comparison_result.csv created")

if __name__ == "__main__":
    dataset_folder = 'dataset'  
    ground_truth_file = 'ground_truth.csv'  

    process_dataset(dataset_folder)
    compare_with_ground_truth('predicted.csv', ground_truth_file)

