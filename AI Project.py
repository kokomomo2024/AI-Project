
import os
import streamlit as st
from PIL import Image
import random
import pandas as pd
from datetime import datetime
from io import BytesIO
import numpy as np
import soundfile as sf
import openai
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ----------------- Configuration -----------------
openai.api_key = os.getenv("OPENAI_API_KEY", "")  # set your OpenAI API key in environment
RESULTS_PATH = "results.xlsx"
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")  # or another model you prefer

st.set_page_config(page_title="AI Project", layout="wide", page_icon=":wrench:")

# ----------------- Helper functions -----------------
def classify_image_text_only(image):
    """Return a text classification (Arabic labels). Placeholder random classifier."""
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

# ----------------- Audio recorder processor -----------------
class RecorderProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []  # list of numpy arrays

    def recv_audio(self, frame):
        audio = frame.to_ndarray()  # shape (frames, channels)
        self.buffer.append(audio)
        return frame

# ----------------- Page styling (soft pink background) -----------------
page_css = """
<style>
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #fff6fb 0%, #fff0f6 100%);
}
.main .block-container { background: rgba(255, 255, 255, 0.66); border-radius: 12px; padding: 1.25rem 1.5rem; }
.header-title { color: #AD1457; font-weight: 800; font-size:34px; margin:0; }
.header-sub { color: #6b2b3b; margin-top:6px; font-weight:600; }
.stButton>button { background: linear-gradient(90deg, #ff9fc0, #ff6fa3); color: white; border-radius: 10px; }
[data-testid="stDataFrameContainer"] { background: rgba(255,255,255,0.7) !important; border-radius: 8px; padding: 0.5rem; }
.signature { text-align: center; margin-top: 1.25rem; font-size: 20px; font-weight: 800; background: -webkit-linear-gradient(#ff5fa8, #ffd166); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)

# ----------------- Header -----------------
col1, col2 = st.columns([0.82, 0.18])
with col1:
    st.markdown('<div style="display:flex;flex-direction:column;">', unsafe_allow_html=True)
    st.markdown('<h1 class="header-title">AI Project — Designed by Mohamed Ashraf</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub"><em>AI-powered defect classification and analysis</em></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    # intentionally left blank (no logo)
    st.write("")

st.markdown("---")

# ----------------- Main layout: Image analysis (text-only) -----------------
left, right = st.columns([2, 1])

with left:
    uploaded_file = st.file_uploader("Upload part image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
        if st.button("Analyze (text only)"):
            with st.spinner("Analyzing..."):
                result_ar = classify_image_text_only(image)
                st.markdown(f"<h2 style='color:#c2185b'>Result: <span style='background:rgba(255,255,255,0.88);padding:6px 10px;border-radius:8px;font-weight:700;'>{result_ar}</span></h2>", unsafe_allow_html=True)
                save_result(uploaded_file.name, result_ar)
                st.success("تم حفظ النتيجة في الملف results.xlsx")
                st.markdown("If you disagree with the result or want more info about any defect type, go to the Voice Chat section below.")

with right:
    st.subheader("Actions")
    df = load_results()
    st.write("Total results:", len(df))
    if not df.empty:
        st.dataframe(df)
        excel_bytes = results_to_excel_bytes(df)
        st.download_button(label="Download results (Excel)", data=excel_bytes, file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if st.button("Refresh results"):
        st.experimental_rerun()
    if st.button("Clear all results"):
        clear_results()
        st.success("All results deleted.")
        st.experimental_rerun()

st.markdown("---")

# ----------------- Voice Chat (available anytime) -----------------
st.header("Voice Chat with AI (Live)")
st.markdown("Start the voice streamer (allow microphone), speak to ask about defects or to challenge a previous classification. Optionally select an image from results to give context to the AI.")

# ensure session state keys
if "recorder" not in st.session_state:
    st.session_state.recorder = None

# webrtc streamer
webrtc_ctx = webrtc_streamer(
    key="voice-chat",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=RecorderProcessor,
    media_stream_constraints={"audio": True, "video": False},
    
    async_processing=True,
)

# attach processor
if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
    st.session_state.recorder = webrtc_ctx.audio_processor

# allow user to choose a past image as context (optional)
df = load_results()
image_names = df["Image Name"].tolist() if not df.empty else []
selected_image = st.selectbox("Select an image from results to provide context (optional)", options=["(no image)"] + image_names)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Transcribe & Ask AI"):
        proc = st.session_state.get("recorder", None)
        if not proc or not hasattr(proc, "buffer") or len(proc.buffer) == 0:
            st.warning("No audio recorded yet. Start the voice streamer and speak a short sentence first.")
        else:
            arr = np.concatenate(proc.buffer, axis=0)
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
            tmp_wav = "temp_recording.wav"
            sf.write(tmp_wav, arr, 48000)
            st.info("Audio recorded. Sending to transcription...")

            # transcribe with Whisper (OpenAI)
            try:
                with open(tmp_wav, "rb") as f:
                    transcription = openai.Audio.transcribe("whisper-1", f)
                    user_text = transcription.get("text", "").strip()
                st.markdown(f"**You said (transcribed):** {user_text}")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                user_text = ""

            proc.buffer = []  # reset buffer

            if user_text:
                # build prompt: if user selected an image, include its last classification
                context_text = ""
                if selected_image != "(no image)":
                    last_rows = df[df["Image Name"] == selected_image]
                    if not last_rows.empty:
                        last_label = last_rows.iloc[-1]["Result"]
                        context_text = f"Previous automated classification for image '{selected_image}' was: '{last_label}'.\n"

                prompt = (
                    f"{context_text}User (in Arabic) says: \"{user_text}\".\n"
                    "You are an expert assistant for diagnosing part defects. If the user is challenging a previous classification, consider the context and decide whether to keep the original label or suggest a new one from: [سليمة, عيب مورد, عيب تجميع]. "
                    "Answer in Arabic. Give a short conclusion (suggested label if applicable) and a brief explanation or guidance. If the user asks for general information about a defect type, explain clearly."
                )

                # call OpenAI ChatCompletion
                try:
                    response = openai.ChatCompletion.create(
                        model=GPT_MODEL,
                        messages=[{"role":"system","content":"You are a helpful Arabic-speaking assistant specialized in diagnosing manufacturing part defects."},
                                  {"role":"user","content":prompt}],
                        max_tokens=500,
                        temperature=0.2,
                    )
                    ai_text = response["choices"][0]["message"]["content"].strip()
                    st.markdown(f"**AI (text):** {ai_text}")
                except Exception as e:
                    st.error(f"AI request failed: {e}")
                    ai_text = ""

                # detect suggested label (heuristic)
                suggested_label = None
                for lbl in ["سليمة", "عيب مورد", "عيب تجميع"]:
                    if lbl in ai_text:
                        suggested_label = lbl
                        break

                if suggested_label:
                    st.success(f"AI suggests: **{suggested_label}**")
                    if st.button("Apply suggested classification"):
                        # save updated classification for the selected image if provided; otherwise save as a new result without image name
                        img_name_to_save = selected_image if selected_image != "(no image)" else f"voice-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        save_result(img_name_to_save, suggested_label)
                        st.success(f"Saved suggested classification: {suggested_label} for '{img_name_to_save}'")

                # text-to-speech reply (gTTS)
                if ai_text:
                    try:
                        tts = gTTS(ai_text, lang="ar")
                        tts_path = "ai_response.mp3"
                        tts.save(tts_path)
                        audio_bytes = open(tts_path, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")
                    except Exception as e:
                        st.error(f"TTS failed: {e}")

with col2:
    st.markdown("**Voice Controls / Tips**")
    st.write("- Start the voice streamer and allow microphone access in the browser.")
    st.write("- Speak clearly. After speaking press **Transcribe & Ask AI**.")
    st.write("- You can select a past image from results to reference it (optional).")
    st.write("- The AI replies in Arabic (text + audio).")

st.markdown("---")
st.markdown('<div class="signature">✨ Designed by Mohamed Ashraf ✨</div>', unsafe_allow_html=True)
