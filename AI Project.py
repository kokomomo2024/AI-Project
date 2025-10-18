import os
import streamlit as st
from PIL import Image
import random
import pandas as pd
from datetime import datetime
from io import BytesIO
from openai import OpenAI

# إعداد مفتاح OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=openai_api_key)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

RESULTS_PATH = "results.xlsx"

st.set_page_config(page_title="AI Project - Mohamed Ashraf", layout="wide", page_icon="🤖")

# ---------------------- الدوال المساعدة ----------------------
def classify_image_text_only(image):
    """تصنيف بسيط (عشوائي مؤقتًا)"""
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

# ---------------------- الواجهة ----------------------
st.title("🤖 AI Project — Designed by Mohamed Ashraf")
st.write("نظام ذكي لتحليل وتصنيف العيوب الصناعية")

tab1, tab2 = st.tabs(["📸 تصنيف الصور", "💬 المحادثة الكتابية مع الذكاء الصناعي"])

# ---------------------- تبويب الصور ----------------------
with tab1:
    uploaded_file = st.file_uploader("حمّل صورة الجزء (jpg أو png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        if st.button("تحليل الصورة"):
            with st.spinner("جاري التحليل..."):
                result_ar = classify_image_text_only(image)
                st.success(f"النتيجة: {result_ar}")
                save_result(uploaded_file.name, result_ar)
    st.markdown("---")
    st.subheader("📄 النتائج السابقة")
    df = load_results()
    if not df.empty:
        st.dataframe(df)
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button("تحميل النتائج (Excel)", buffer, "results.xlsx")

# ---------------------- تبويب المحادثة ----------------------
with tab2:
    st.subheader("تحدث مع الذكاء الصناعي كتابةً 💬")
    user_input = st.text_area("اكتب سؤالك هنا بالعربية:")
    if st.button("إرسال"):
        if not user_input.strip():
            st.warning("من فضلك اكتب شيئًا أولًا.")
        else:
            with st.spinner("جاري الرد..."):
                try:
                    response = client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=[
                            {"role": "system", "content": "أنت مساعد ناطق بالعربية متخصص في تصنيف العيوب الصناعية."},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=500,
                        temperature=0.4,
                    )
                    ai_text = response.choices[0].message.content.strip()
                    st.markdown(f"**رد الذكاء الصناعي:** {ai_text}")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء الاتصال بالذكاء الصناعي: {e}")

st.markdown("---")
st.markdown('<div style="text-align:center;font-weight:bold;">✨ Designed by Mohamed Ashraf ✨</div>', unsafe_allow_html=True)
