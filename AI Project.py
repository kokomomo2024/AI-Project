# ai_trainer_app.py
import os
import streamlit as st
from PIL import Image
import random
import pandas as pd
from datetime import datetime
from io import BytesIO
from gtts import gTTS
import base64
from openai import OpenAI

# ----------------- Configuration -----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
RESULTS_PATH = "results.xlsx"
TRAINING_DATA_PATH = "training_data.xlsx"
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="AI Project — Trainer", layout="wide", page_icon=":wrench:")

# ----------------- Helpers -----------------
def image_to_base64_bytes(file) -> str:
    if file is None:
        return ""
    file.seek(0)
    return base64.b64encode(file.read()).decode("utf-8")

def base64_to_image(b64: str):
    if not b64:
        return None
    return Image.open(BytesIO(base64.b64decode(b64)))

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

# Training data functions
def load_training_data(path=TRAINING_DATA_PATH):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            # if file corrupted, return empty template
            return pd.DataFrame(columns=[
                "اسم العيب", "الوصف", "مستوى الخطورة", "تم عليه تجميع",
                "العبوة سليمة", "خطوات التجميع سليمة",
                "image_good_b64", "image_defect_b64",
                "ai_guess", "user_correction", "correction_reason", "timestamp"
            ])
    else:
        return pd.DataFrame(columns=[
            "اسم العيب", "الوصف", "مستوى الخطورة", "تم عليه تجميع",
            "العبوة سليمة", "خطوات التجميع سليمة",
            "image_good_b64", "image_defect_b64",
            "ai_guess", "user_correction", "correction_reason", "timestamp"
        ])

def save_training_row(row_dict, path=TRAINING_DATA_PATH):
    df = load_training_data(path)
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_excel(path, index=False)

# Build few-shot examples from recent corrected rows
def build_few_shot_examples(n_examples=3):
    df = load_training_data()
    examples = []
    if df.empty:
        return examples
    df_valid = df.dropna(subset=["user_correction"]).tail(n_examples)
    for _, r in df_valid.iterrows():
        ex = {
            "description": str(r.get("الوصف", "")),
            "ai_guess": str(r.get("ai_guess", "")),
            "correction": str(r.get("user_correction", "")),
            "reason": str(r.get("correction_reason", ""))
        }
        examples.append(ex)
    return examples

# Compose system prompt with few-shot
def compose_system_prompt():
    base = (
        "أنت مساعد تشخيص عيوب صناعية يتعلم من المدرب. "
        "عند تحليلك لحالة، أعطِ نتيجة مُهيكلة بصيغة JSON فقط "
        "مع الحقول: type, severity, confidence (0-100), reason, recommendation. "
        "ثم اقترح سؤال توضيحي واحد إذا الثقة أقل من 90%."
    )
    examples = build_few_shot_examples()
    if examples:
        base += "\n\nأمثلة للتعلم (few-shot):\n"
        for i, ex in enumerate(examples, 1):
            base += f"\nمثال {i}:\n"
            base += f"الوصف: {ex['description']}\n"
            base += f"تحليل النموذج: {ex['ai_guess']}\n"
            base += f"تصحيح المدرب: {ex['correction']}\n"
            base += f"السبب: {ex['reason']}\n"
    return base

# ----------------- Page styling (kept from original) -----------------
page_css = """
<style>
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #fff6fb 0%, #fff0f6 100%);
}
.main .block-container {
  background: rgba(255, 255, 255, 0.88);
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
  font-size: 18px;
  font-weight: 700;
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
    st.markdown('<h1 class="header-title">AI Project — Trainer by Mohamed Ashraf</h1>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub"><em>Interactive AI trainer: human-in-the-loop defect classification</em></div>', unsafe_allow_html=True)
with col2:
    st.write("")

st.markdown("---")

# ----------------- Main layout -----------------
left, right = st.columns([2, 1])

with left:
    st.header("🔍 تحليل سريع وبدء التدريب")
    st.write("ارفع صورة الجزء المعيب (أو اجمع بيانات كاملة عبر زر '➕ إضافة عيب جديد').")
    uploaded_file = st.file_uploader("Upload single part image (jpg, jpeg, png) — optional", type=["jpg", "jpeg", "png"], key="single_upload")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

    # Quick analyze with manual description (if user prefers)
    st.subheader("تحليل سريع — اكتب وصف مختصر")
    quick_desc = st.text_area("وصف الحالة (اختياري):", placeholder="أدخل وصفًا قصيرًا للحالة إن وُجد")
    if st.button("Analyze (AI) — Quick", key="quick_analyze"):
        if not quick_desc.strip() and not uploaded_file:
            st.warning("أدخل وصف أو ارفع صورة لتحليل أسرع.")
        else:
            with st.spinner("Generating AI analysis..."):
                system_prompt = compose_system_prompt()
                user_msg = f"تحليل الحالة الوصفية التالية:\n{quick_desc}\n(ملاحظة: صورة مرفوعة؟ {'نعم' if uploaded_file else 'لا'})"
                try:
                    response = client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg}
                        ],
                        max_tokens=700,
                        temperature=0.2,
                    )
                    ai_text = response.choices[0].message.content.strip()
                    st.markdown("**🤖 AI (structured) — response:**")
                    st.code(ai_text, language="json")
                    # Save quick result to results.xlsx
                    save_result(uploaded_file.name if uploaded_file else f"desc_{datetime.now().strftime('%Y%m%d%H%M%S')}", ai_text)
                    st.success("✅ تم حفظ النتيجة في results.xlsx")
                    # TTS (optional)
                    try:
                        tts = gTTS(ai_text, lang="ar")
                        tts_path = "ai_response.mp3"
                        tts.save(tts_path)
                        audio_bytes = open(tts_path, "rb").read()
                        st.audio(audio_bytes, format="audio/mp3")
                    except Exception:
                        # ignore TTS errors
                        pass
                    st.session_state["last_ai_text"] = ai_text
                    st.session_state["last_description"] = quick_desc
                except Exception as e:
                    st.error(f"Error communicating with OpenAI: {e}")

    st.markdown("---")
    st.header("➕ إضافة عيب جديد (سليم vs تالف) — للتدريب التفصيلي")

    if st.button("➕ إضافة عيب جديد"):
        st.session_state["add_mode"] = True

    if "add_mode" in st.session_state and st.session_state["add_mode"]:
        with st.form("add_defect_form"):
            col1a, col2a = st.columns(2)
            with col1a:
                defect_name = st.text_input("اسم العيب:")
                severity = st.selectbox("مستوى الخطورة:", ["منخفض", "متوسط", "مرتفع"])
                assembly_done = st.selectbox("هل تم عليه عملية تجميع؟", ["نعم", "لا", "غير معروف"])
            with col2a:
                package_ok = st.selectbox("هل العبوة سليمة؟", ["نعم", "لا", "غير معروف"])
                steps_ok = st.selectbox("هل تم تنفيذ خطوات التجميع كاملة؟", ["نعم", "لا", "غير معروف"])
                description = st.text_area("وصف التفاصيل (مهم):", height=120)

            st.markdown("**رفع الصور**")
            col3, col4 = st.columns(2)
            with col3:
                good_img = st.file_uploader("صورة الجزء السليم", type=["jpg", "jpeg", "png"], key="good_img_upload")
                if good_img:
                    st.image(Image.open(good_img), caption="الجزء السليم", use_container_width=True)
            with col4:
                defect_img = st.file_uploader("صورة الجزء المعيب", type=["jpg", "jpeg", "png"], key="defect_img_upload")
                if defect_img:
                    st.image(Image.open(defect_img), caption="الجزء المعيب", use_container_width=True)

            submit_add = st.form_submit_button("💾 حفظ العيب وتحليل AI")

            if submit_add:
                if not description.strip():
                    st.warning("يرجى كتابة وصف مفصل للحالة حتى يتمكن الـ AI من التحليل الجيد.")
                else:
                    # convert images to base64
                    good_b64 = image_to_base64_bytes(good_img) if 'good_img' in locals() else ""
                    defect_b64 = image_to_base64_bytes(defect_img) if 'defect_img' in locals() else ""

                    # Create prompt with few-shot examples
                    system_prompt = compose_system_prompt()
                    user_msg = (
                        f"حالة جديدة للتحليل:\n"
                        f"اسم العيب: {defect_name}\n"
                        f"الوصف: {description}\n"
                        f"مستوى الخطورة: {severity}\n"
                        f"تم عليه تجميع: {assembly_done}\n"
                        f"العبوة سليمة: {package_ok}\n"
                        f"خطوات التجميع سليمة: {steps_ok}\n"
                        f"(يوجد صورتين: جزء سليم وجزء تالف — الصور مخزنة محليًا في سجلات التدريب.)\n"
                        f"أجب بصيغة JSON فقط: {{type, severity, confidence (0-100), reason, recommendation}}.\n"
                        f"إذا كانت الثقة أقل من 90% اقترح سؤال توضيحي واحد."
                    )

                    with st.spinner("جاري تحليل AI..."):
                        try:
                            response = client.chat.completions.create(
                                model=GPT_MODEL,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_msg}
                                ],
                                max_tokens=700,
                                temperature=0.2,
                            )
                            ai_text = response.choices[0].message.content.strip()
                        except Exception as e:
                            st.error(f"خطأ في الاتصال بـ OpenAI: {e}")
                            ai_text = "Error: AI did not respond."

                    # Save to training data with empty correction for now
                    row = {
                        "اسم العيب": defect_name,
                        "الوصف": description,
                        "مستوى الخطورة": severity,
                        "تم عليه تجميع": assembly_done,
                        "العبوة سليمة": package_ok,
                        "خطوات التجميع سليمة": steps_ok,
                        "image_good_b64": good_b64,
                        "image_defect_b64": defect_b64,
                        "ai_guess": ai_text,
                        "user_correction": "",
                        "correction_reason": "",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_training_row(row)
                    st.success("✅ تم حفظ حالة التدريب مع نتيجة AI. الآن يمكنك تصحيحها في قسم 'حوار التدريب' أدناه.")
                    st.session_state["last_saved_index"] = True

    st.markdown("---")
    st.header("🗣️ حوار التدريب — درِّب الـAI يدويًا (مدرّب ↔ متدرب)")

    # Show latest AI entries that need correction (user_correction empty) or all
    df_train = load_training_data()
    pending = df_train[df_train["user_correction"].isna() | (df_train["user_correction"] == "")]
    st.write(f"عدد الحالات المسجلة: {len(df_train)} — حالات بحاجة لتصحيح: {len(pending)}")

    # Let user pick a pending case to correct (or pick any case)
    choice_mode = st.radio("اختر عرض الحالات:", ["الحالات التي تحتاج تصحيح", "كل الحالات"], index=0)
    if choice_mode == "الحالات التي تحتاج تصحيح":
        show_df = pending.reset_index(drop=True)
    else:
        show_df = df_train.reset_index(drop=True)

    if not show_df.empty:
        # display table without big image columns
        st.dataframe(show_df.drop(columns=[c for c in ["image_good_b64", "image_defect_b64"] if c in show_df.columns]))
        sel_idx = st.number_input("اختر رقم السطر للتصحيح (index):", min_value=0, max_value=max(0, len(show_df)-1), value=0, step=1)
        sel_row = show_df.loc[sel_idx]
        st.markdown("### الحالة المحددة")
        st.write(f"**اسم العيب:** {sel_row.get('اسم العيب','')}")
        st.write(f"**الوصف:** {sel_row.get('الوصف','')}")
        st.write(f"**تحليل AI السابق:**")
        st.code(sel_row.get("ai_guess",""))

        # show images if present
        col_a, col_b = st.columns(2)
        with col_a:
            img_good_b64 = sel_row.get("image_good_b64", "")
            if img_good_b64:
                st.image(base64.b64decode(img_good_b64), caption="الجزء السليم", use_column_width=True)
        with col_b:
            img_def_b64 = sel_row.get("image_defect_b64", "")
            if img_def_b64:
                st.image(base64.b64decode(img_def_b64), caption="الجزء المعيب", use_column_width=True)

        # Correction input
        st.markdown("#### أدخل تصحيحك كمدرّب")
        user_correction = st.text_input("التصنيف الصحيح (مثال: عيب تجميع)", key="user_correction_input")
        correction_reason = st.text_area("سبب التصحيح (اشرح باختصار):", key="correction_reason_input")

        if st.button("💾 حفظ التصحيح والتعليم"):
            # update the real dataframe (we used show_df as a view)
            real_df = load_training_data()
            # find the actual row by matching timestamp or description
            # fallback: match first row with same ai_guess & description & empty user_correction
            mask = (real_df["ai_guess"] == sel_row.get("ai_guess")) & (real_df["الوصف"] == sel_row.get("الوصف"))
            idxs = real_df[mask].index.tolist()
            target_idx = idxs[0] if idxs else None
            if target_idx is None:
                # more robust fallback: try to match by timestamp if present
                ts = sel_row.get("timestamp", "")
                if ts:
                    matched = real_df[real_df["timestamp"] == ts]
                    if not matched.empty:
                        target_idx = matched.index[0]
            if target_idx is None:
                st.error("لم أتمكن من تحديد السطر بدقّة — تأكد من اختيار السطر الصحيح.")
            else:
                real_df.at[target_idx, "user_correction"] = user_correction
                real_df.at[target_idx, "correction_reason"] = correction_reason
                real_df.at[target_idx, "timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                real_df.to_excel(TRAINING_DATA_PATH, index=False)
                st.success("✅ تم حفظ التصحيح. سيتم استخدام هذه الأمثلة لتحسين ردود AI لاحقًا (few-shot).")

    else:
        st.info("لا توجد حالات للعرض في الوقت الحالي.")

with right:
    st.subheader("Actions — إدارة ونتائج")
    df = load_results()
    st.write("Total results saved:", len(df))
    if not df.empty:
        st.dataframe(df)
        excel_bytes = results_to_excel_bytes(df)
        st.download_button(
            label="📥 Download results (Excel)",
            data=excel_bytes,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    # Training data actions
    st.markdown("---")
    st.subheader("Training data")
    tdf = load_training_data()
    st.write("Total training rows:", len(tdf))
    if not tdf.empty:
        # show table without image columns
        st.dataframe(tdf.drop(columns=[c for c in ["image_good_b64", "image_defect_b64"] if c in tdf.columns]))
        # download button for training data
        buffer = BytesIO()
        tdf.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button("📥 Download training_data.xlsx", data=buffer.getvalue(), file_name="training_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if st.button("🔄 Refresh all"):
        st.experimental_rerun()

    if st.button("🗑️ Delete all results"):
        clear_results()
        st.success("✅ All results have been deleted successfully.")
        st.experimental_rerun()

st.markdown("---")
st.markdown('<div class="signature">✨ Designed by Mohamed Ashraf — AI Trainer ✨</div>', unsafe_allow_html=True)
