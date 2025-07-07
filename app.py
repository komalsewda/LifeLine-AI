
import streamlit as st
from palmistry import (
    extract_lines,
    analyze_palm_features,
    generate_gemini_reading,
    translate_to_hindi,
    generate_speech,
)
from utils import load_images_from_folder
import tempfile
import os

# ──────────── Config ──────────────
st.set_page_config(page_title="PalmReader AI", layout="centered")
st.title("🖐️ PalmReader AI")
st.markdown("Upload your palm image or select one from folder `001/` to get a real-time palm line analysis.")

# ──────────── Globals ─────────────
image_folder = "001"
selected_filename = None

# ──────────── Language & Voice ─────────────
lang = st.radio("🌍 Choose Reading Language", ["English", "Hindi"], horizontal=True)
speak = st.checkbox("🔊 Generate Voice Reading (English Only)")

# ──────────── Upload Image ─────────────
st.header("📤 Upload Your Palm Image")
image_file = st.file_uploader("Choose a palm image...", type=["jpg", "jpeg", "png"])

if image_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_file.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="📷 Uploaded Palm", use_container_width=True)

    edges = extract_lines(tmp_path)
    features = analyze_palm_features(edges)

    st.subheader("🧠 Extracted Palm Lines")
    st.image(edges, caption="🧠 Detected Lines", channels="GRAY", use_container_width=True)

    st.subheader("📊 Detected Palm Line Features")
    st.markdown(f"**Total main lines detected:** {len(features)}")
    for i, f in enumerate(features):
        st.markdown(f"**Line {i+1}**: Length = `{f['length']}`, Area = `{f['area']}`, Points = `{f['points']}`")

    st.subheader("🔮 Personalized Reading")
    with st.spinner("Generating Gemini reading..."):
        reading = generate_gemini_reading(features)

        if lang == "Hindi":
            reading = translate_to_hindi(reading)

        st.markdown(reading)

        if speak and lang == "English":
            mp3_path = generate_speech(reading)
            if mp3_path:
                with open(mp3_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")
            else:
                st.error("❌ Failed to generate speech.")

# ──────────── Chatbot ─────────────
if "reading_context" not in st.session_state:
    st.session_state.reading_context = ""

if image_file:
    st.markdown("## 💬 Ask Palm Bot")
    st.markdown("Ask about your **career**, **marriage**, **health**, or anything related to your palm reading.")

    user_question = st.text_input("🗨️ Your Question:")
    if st.button("🔍 Ask Bot") and user_question:
        with st.spinner("Gemini is thinking..."):
            full_prompt = f"""
You are a palmistry expert AI.

Below is the palm reading of a user:
{reading}

Now answer this specific question from the user: "{user_question}"

✋ Guidelines:
- Give a clear, specific answer. 
- If the user asks about **marriage**, estimate the **likely age range** or **specific year** when marriage is most probable (e.g., "around age 24–25", "in late 2026").
- Use the palm features like Life Line, Heart Line, and Fate Line to justify your answer.
- Be realistic and grounded — avoid vague phrases like "whenever you're ready" or "fate will decide".
- Don’t be overly spiritual or magical. Be wise and practical.
- Keep the answer short: 1–2 paragraphs maximum.

User’s current age is: 20
"""

            try:
                import google.generativeai as genai
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(full_prompt)
                st.markdown(f"**🧞 Answer:**\n\n{response.text}")
                st.session_state.reading_context += f"\nUser: {user_question}\nBot: {response.text}\n"
            except Exception as e:
                st.error(f"❌ Chat error: {str(e)}")

# ──────────── Sample Folder Load ─────────────
st.markdown("---")
st.header("🗂️ Or Select an Image from Folder `001/`")

if os.path.exists(image_folder):
    images, paths = load_images_from_folder(image_folder)
    if images:
        filenames = [os.path.basename(p) for p in paths]
        selected_filename = st.selectbox("Choose an image:", filenames)

        selected_index = filenames.index(selected_filename)
        selected_image_path = paths[selected_index]

        if st.button("📁 Load Selected Palm"):
            st.image(selected_image_path, caption=f"📸 Selected: {selected_filename}", use_container_width=True)

            edges = extract_lines(selected_image_path)
            features = analyze_palm_features(edges)

            st.subheader("🧠 Extracted Palm Lines")
            st.image(edges, caption="🧠 Detected Lines", channels="GRAY", use_container_width=True)

            st.subheader("📊 Detected Palm Line Features")
            st.markdown(f"**Total main lines detected:** {len(features)}")
            for i, f in enumerate(features):
                st.markdown(f"**Line {i+1}**: Length = `{f['length']}`, Area = `{f['area']}`, Points = `{f['points']}`")

            st.subheader("🔮 Personalized Reading")
            with st.spinner("Generating Gemini reading..."):
                reading = generate_gemini_reading(features)

                if lang == "Hindi":
                    reading = translate_to_hindi(reading)

                st.markdown(reading)

                if speak and lang == "English":
                    mp3_path = generate_speech(reading)
                    if mp3_path:
                        with open(mp3_path, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")
                    else:
                        st.error("❌ Failed to generate speech.")
    else:
        st.warning("⚠️ No images found in folder `001/`. Add some JPG/PNG images there.")
else:
    st.error("❌ Folder `001/` not found. Please create it and place palm images inside.")

# ──────────── Footer ─────────────
st.markdown("---")
st.markdown("✨ Made with ❤️ by Komal | Powered by Gemini + Palmistry + Voice")
