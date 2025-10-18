import os
import streamlit as st
from PIL import Image
import random
import pandas as pd
from datetime import datetime
from io import BytesIO
from gtts import gTTS
from openai import OpenAI  # ✅ new import

# ----------------- Configuration -----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))  # ✅ new API usage
RESULTS_PATH = "results.xlsx"
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="AI Project", layout="wide", page_icon=":wrench:")

# ----------------- Helper functions -----------------
def classify_image_text_only(image):
    labels_ar = ["سليمة", "عيب مورد", "عيب تجميع"]
    return random.choice(labels_ar)

def load_results(path=RESULTS_PATH):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.DataFrame(columns=["Image Name", "Result", "Time"])
    else:
        return pd.DataFrame(columns=["Image Name", "Result", "Time"])

def save_result(filename, result, path=RESULTS_PATH):
    df = load_results(path)
    new_row = {
        "Image Name": filename,
        "Result": result,
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(path, index=False)
    return df

def results_to_excel_bytes(df):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

def clear_results(path=RESULTS_PATH):
    if os.path.exists(path):
        os.remove(path)

# ----------------- Page styling -----------------
page_css = """
<style>
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #fff6fb 0%, #fff0f6 100%);
}
.main .block-container {
  background: rgba(255, 255, 255, 0.66);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
}
.header-title { color: #AD1457; font-weight: 800; font-size:34px; margin:0; }
.header-sub { color: #6b2b3b; margin-top:6px; font-weight:600; }
.stButton>button {
  background: linear-gradient(90deg, #ff9fc0, #ff6fa3);
  color: white;
  border-radius: 10px;
}
[data-testid="stDataFrameContainer"] {
  background: rgba(255,255,255,0.7) !important;
  border-radius: 8px;
  padding: 0.5rem;
}
.signature {
  text-align: center;
  margin-top: 1.25rem;
  font-size: 20px;
  font-weight: 800;
  background: -webkit-linear-gradient(#ff5fa8, #ffd166);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)

# ----------------- Header -----------------
col1, col2 = st.columns([0.82, 0.18])
with col1:
    st.markdown('<h1 class="header-title">AI Project — Designed by Mohamed Ashraf</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub"><em>AI-powered defect classification and analysis</em></div>', unsafe_allow_html=True)
with col2:
    st.write("")

st.markdown("---")

# ----------------- Main layout -----------------
left, right = st.columns([2, 1])

with left:
    uploaded_file = st.file_uploader("Upload part image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Analyze (text only)"):
            with st.spinner("Analyzing..."):
                result_ar = classify_image_text_only(image)
                st.markdown(
                    f"<h2 style='color:#c2185b'>Result: "
                    f"<span style='background:rgba(255,255,255,0.88);padding:6px 10px;border-radius:8px;font-weight:700;'>{result_ar}</span></h2>",
                    unsafe_allow_html=True,
                )
                save_result(uploaded_file.name, result_ar)
                st.success("Result saved to results.xlsx")

with right:
    st.subheader("Actions")
    df = load_results()
    st.write("Total results:", len(df))
    if not df.empty:
        st.dataframe(df)
        excel_bytes = results_to_excel_bytes(df)
        st.download_button(
            label="📥 Download results (Excel)",
            data=excel_bytes,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    if st.button("🔄 Refresh results"):
        st.experimental_rerun()
    if st.button("🗑️ Delete all results"):
        clear_results()
        st.success("✅ All results have been deleted successfully.")
        st.experimental_rerun()

st.markdown("---")

# ----------------- Text-based Chat with AI -----------------
st.header("💬 Chat with AI")
st.markdown("Type your question or comment below, and the AI will reply in Arabic (text + voice).")

user_input = st.text_area("✏️ Type your message:", placeholder="For example: What is the difference between supplier defect and assembly defect?")

if st.button("Send to AI"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Generating AI response..."):
            try:
                # ✅ new API call
                response = client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful Arabic-speaking assistant specialized in diagnosing manufacturing part defects."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=500,
                    temperature=0.4,
                )
                ai_text = response.choices[0].message.content.strip()
                st.markdown(f"**🤖 AI Response:** {ai_text}")

                # ✅ Convert AI reply to Arabic audio
                try:
                    tts = gTTS(ai_text, lang="ar")
                    tts_path = "ai_response.mp3"
                    tts.save(tts_path)
                    audio_bytes = open(tts_path, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception as e:
                    st.error(f"Text-to-speech failed: {e}")

            except Exception as e:
                st.error(f"Error communicating with OpenAI: {e}")

st.markdown("---")
st.markdown('<div class="signature">✨ Designed by Mohamed Ashraf ✨</div>', unsafe_allow_html=True)
