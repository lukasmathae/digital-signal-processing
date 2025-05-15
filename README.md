# 📦 AMK/Weight Detection System with Visual Prompting

This project automatically extracts the **AMK number** (client ID) and **weight** from parcel images using a hybrid system that combines traditional OCR and **GPT-4 Vision (Visual Prompting)** as a fallback.

---

## 🔍 What Is Visual Prompting?

**Visual Prompting** is the process of using a visual model like GPT-4 Vision to interpret an image based on a carefully crafted **natural language prompt**.

In this system, if OCR fails to detect required information (due to low quality, small text, or foreign labels), GPT-4 Vision is prompted with a sentence like:

> “This is a parcel image. Please extract the AMK number (client ID, like AMK02149) and the weight (e.g., 0.28kg). If AMK is not found, return 'N/A'.”

This allows GPT-4V to **visually understand and extract** information even from challenging or inconsistent layouts.

---

## ⚙️ System Workflow

1. **Input**: A folder of parcel images
2. **OCR Phase**:
    - Run `Tesseract OCR` on each image
    - Use regular expressions to find:
      - AMK number: `AMK\d{4,6}`
      - Weight: `(\d+(\.\d+)?)\s?kg`
3. **Fallback to GPT-4 Vision**:
    - If OCR fails to find valid results, send image to GPT-4 Vision API
    - Receive extracted values using natural language response
4. **Output**: A `results.csv` file with:
    - Filename
    - Predicted AMK
    - Predicted Weight
5. **Evaluation** (optional): 
    - Compare results with `ground_truth.csv`
    - Measure prediction accuracy

---

## 🐍 Files Included

| File | Description |
|------|-------------|
| `detect_amk_weight.py` | Main script: handles image processing, OCR, GPT-4 fallback, and saves predictions |
| `ground_truth.csv` | Manually verified results to compare against model predictions |
| `results.csv` | Output CSV with predicted AMK and weight per image |
| `.env` | (not included in repo) Contains your OpenAI API key |
| `.gitignore` | Ensures `.env` and `.venv` are excluded from Git history |

---

## 🧠 Technologies Used

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenAI GPT-4 Vision API](https://platform.openai.com/docs/guides/vision)
- Pandas
- dotenv

