
import cv2
import numpy as np
import google.generativeai as genai
from gtts import gTTS
import tempfile

# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyA3JIUwgdVVLvlWvAPX3dywFuCHZaNlEfI")  # Replace before running

# ───── 1. Enhanced Palm Line Extraction ─────
def extract_lines(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("❌ Image not found!")

    img = cv2.resize(img, (600, 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ⚡ Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ⚡ Sharpen image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)

    # ⚡ Smooth slightly
    blur = cv2.GaussianBlur(sharp, (5, 5), 0)

    # ⚡ Detect edges (fine tuning thresholds)
    edges = cv2.Canny(blur, 30, 100)
    return edges

# ───── 2. Feature Detection ─────
def analyze_palm_features(edge_image):
    contours, _ = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    features = []

    for cnt in contours:
        length = cv2.arcLength(cnt, False)
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * length, True)

        if length > 100 and area > 50:
            features.append({
                "length": round(length, 2),
                "area": round(area, 2),
                "points": len(approx)
            })

    return features

# ───── 3. Format for Gemini ─────
def format_features_for_prompt(features):
    prompt_lines = "These are the palm line features extracted from the hand image:\n"
    for i, f in enumerate(features[:7]):
        prompt_lines += f"Line {i+1}: Length = {f['length']}, Area = {f['area']}, Points = {f['points']}\n"
    return prompt_lines

# ───── 4. Gemini Reading ─────
def generate_gemini_reading(features):
    model = genai.GenerativeModel("gemini-1.5-flash")
    feature_description = format_features_for_prompt(features)

    prompt = f"""
You are an expert palm reader.

{feature_description}

Now write a palm reading in simple, easy-to-understand language.

Instructions:
- Use short, simple sentences.
- Avoid complex or magical language.
- Clearly explain the Life Line, Head Line, Heart Line, Fate Line, and Sun Line.
- Talk about the person's childhood, present, and future.
- Describe personality traits.
- Give realistic advice on health, relationships, and career.

Make it friendly and human-like. Write 3 to 4 paragraphs.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

# ───── 5. Hindi Translation ─────
def translate_to_hindi(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Translate the following English text to Hindi in a simple, friendly tone:\n\n{text}"

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Hindi translation error: {str(e)}"

# ───── 6. Voice Output ─────
def generate_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception:
        return None
