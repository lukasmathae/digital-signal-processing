import cv2
import pytesseract
import openai
import os
import pandas as pd
import base64
from dotenv import load_dotenv

# === Load API key from .env ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Configuration ===
IMAGE_FOLDER = "images"
OUTPUT_CSV = "results.csv"

# === Visual prompt for GPT-4V ===
VISUAL_PROMPT = (
    "Extract the following from this parcel image:\n"
    "1. AMK number (e.g., AMK02149). If missing, respond AMK: N/A\n"
    "2. Weight in kilograms (e.g., 0.28).\n"
    "Return in this format:\nAMK: ...\nWeight: ..."
)

# === OCR extractor ===
def extract_with_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_text = pytesseract.image_to_string(gray)

    # Try to extract AMK
    amk = "N/A"
    for word in ocr_text.split():
        if word.upper().startswith("AMK"):
            amk = word.upper()
            break

    # Try to extract weight
    weight = "N/A"
    for line in ocr_text.splitlines():
        if "kg" in line.lower():
            parts = line.split()
            for part in parts:
                if part.replace('.', '', 1).isdigit():
                    weight = part
                    break
    return amk, weight

# === GPT-4V extractor ===
def extract_with_gpt4v(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()

    b64_image = base64.b64encode(image_data).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISUAL_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ],
        max_tokens=500
    )

    result = response.choices[0].message.content
    amk = "N/A"
    weight = "N/A"
    for line in result.splitlines():
        if "amk" in line.lower():
            amk = line.split(":")[-1].strip().upper()
        if "weight" in line.lower():
            weight = line.split(":")[-1].strip().replace("kg", "").strip()
    return amk, weight

# === Main process ===
def process_images():
    results = []
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filepath = os.path.join(IMAGE_FOLDER, filename)
        image = cv2.imread(filepath)

        # Step 1: OCR
        amk, weight = extract_with_ocr(image)

        # Step 2: Fallback to GPT-4V
        if amk == "N/A" or weight == "N/A":
            print(f"🔁 OCR failed for {filename}, using GPT-4V...")
            amk, weight = extract_with_gpt4v(filepath)
        else:
            print(f"✅ OCR succeeded for {filename}")

        results.append({
            "filename": filename,
            "amk": amk,
            "weight": weight
        })

    # Save final result
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Results saved to {OUTPUT_CSV}")

# === Run the script ===
if __name__ == "__main__":
    process_images()
