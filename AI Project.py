# ai_trainer_streamlit.py
import os
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
from io import BytesIO
from gtts import gTTS
import uuid
from openai import OpenAI

# ---------------- Config ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# Paths
TRAINING_EXCEL = "training_data.xlsx"
RESULTS_EXCEL = "results.xlsx"
IMAGES_DIR = "images"

os.makedirs(IMAGES_DIR, exist_ok=True)

st.set_page_config(page_title="AI Trainer — Mohamed Ashraf", layout="wide", page_icon=":wrench:")

# ---------------- Helpers ----------------
def load_training_data(path=TRAINING_EXCEL):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            # corrupted -> return empty template
            return pd.DataFrame(columns=[
                "id","defect_name","description","severity","assembly_done",
                "package_ok","steps_ok","img_good","img_defect",
                "ai_guess","user_correction","correction_reason","timestamp"
            ])
    else:
        return pd.DataFrame(columns=[
            "id","defect_name","description","severity","assembly_done",
            "package_ok","steps_ok","img_good","img_defect",
            "ai_guess","user_correction","correction_reason","timestamp"
        ])

def save_training_data_row(row, path=TRAINING_EXCEL):
    df = load_training_data(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(path, index=False)

def save_result_row(filename, result, path=RESULTS_EXCEL):
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.DataFrame(columns=["Image Name","Result","Time"])
    else:
        df = pd.DataFrame(columns=["Image Name","Result","Time"])
    new_row = {"Image Name": filename, "Result": result, "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(path, index=False)

def build_few_shot_examples(n=3):
    df = load_training_data()
    examples = []
    if df.empty:
        return examples
    df_valid = df.dropna(subset=["user_correction"])
    if df_valid.empty:
        return examples
    for _, r in df_valid.tail(n).iterrows():
        examples.append({
            "description": str(r.get("description","")),
            "ai_guess": str(r.get("ai_guess","")),
            "correction": str(r.get("user_correction","")),
            "reason": str(r.get("correction_reason",""))
        })
    return examples

def compose_system_prompt():
    base = (
        "أنت مساعد تشخيص عيوب صناعية باللغة العربية. اعطِ إجابة مُهيكلة (JSON) للحالة تحت الحقول: "
        "type (سليمة / عيب مورد / عيب تجميع / تلف شحن / غير مؤكد)، "
        "severity (منخفض/متوسط/مرتفع)، "
        "confidence (0-100)، "
        "reason (سطرين)، "
        "recommendation (استبدال/إصلاح/قبول). "
        "إذا كانت الثقة أقل من 90 اقترح سؤال توضيحي واحد فقط.\n\n"
    )
    examples = build_few_shot_examples(3)
    if examples:
        base += "أمثلة للتعلّم (few-shot):\n"
        for i, ex in enumerate(examples,1):
            base += f"\nمثال{i}:\nالوصف: {ex['description']}\nتحليل النموذج: {ex['ai_guess']}\nتصحيح المدرب: {ex['correction']}\nالسبب: {ex['reason']}\n"
    return base

def save_uploaded_image(file, prefix="img"):
    if file is None:
        return ""
    # create unique filename
    ext = os.path.splitext(file.name)[1].lower()
    uid = uuid.uuid4().hex
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d')}_{uid}{ext}"
    path = os.path.join(IMAGES_DIR, fname)
    # save file
    img = Image.open(file)
    img.save(path)
    return path

# ---------------- UI ----------------
st.title("🧠 AI Trainer — تدريب الذكاء الصناعي على تصنيف العيوب")
st.markdown("رفع صورتين (سليم + تالف)، أجب على الأسئلة، اضغط 'إضافة العيب للتدريب' وسيحلل AI وسجلّك سيتم حفظه في Excel.")

left, right = st.columns([2,1])

with left:
    st.header("➕ إضافة عيب جديد (سليم vs تالف)")

    with st.form("add_defect"):
        defect_name = st.text_input("اسم العيب (مثال: كسر حافة، عيب لحام):")
        severity = st.selectbox("مستوى الخطورة:", ["منخفض","متوسط","مرتفع"])
        assembly_done = st.selectbox("هل تم عليه عملية تجميع؟", ["نعم","لا","غير معروف"])
        package_ok = st.selectbox("هل العبوة سليمة؟", ["نعم","لا","غير معروف"])
        steps_ok = st.selectbox("هل تم التأكد من خطوات التجميع؟", ["نعم","لا","غير معروف"])
        description = st.text_area("وصف مفصل للحالة (مهم):", height=120)

        st.markdown("**رفع الصور**")
        col1, col2 = st.columns(2)
        with col1:
            good_img_file = st.file_uploader("صورة الجزء السليم", type=["jpg","jpeg","png"], key="good_img")
            if good_img_file:
                st.image(Image.open(good_img_file), caption="الجزء السليم", use_column_width=True)
        with col2:
            defect_img_file = st.file_uploader("صورة الجزء المعيب", type=["jpg","jpeg","png"], key="defect_img")
            if defect_img_file:
                st.image(Image.open(defect_img_file), caption="الجزء المعيب", use_column_width=True)

        submit = st.form_submit_button("💾 إضافة العيب للتدريب وتحليل AI")

        if submit:
            if not description.strip():
                st.warning("الرجاء كتابة وصف مفصل للحالة قبل الحفظ.")
            else:
                # save images to disk
                img_good_path = save_uploaded_image(good_img_file, prefix="good") if good_img_file else ""
                img_defect_path = save_uploaded_image(defect_img_file, prefix="defect") if defect_img_file else ""

                # prepare prompt & call AI
                system_prompt = compose_system_prompt()
                user_msg = (
                    f"حالة جديدة:\nاسم العيب: {defect_name}\nالوصف: {description}\n"
                    f"مستوى الخطورة: {severity}\nتم عليه تجميع: {assembly_done}\n"
                    f"العبوة سليمة: {package_ok}\nخطوات التجميع سليمة: {steps_ok}\n"
                    f"(يوجد صورتان محفوظتان محليًا: {os.path.basename(img_good_path)} و {os.path.basename(img_defect_path)})\n"
                    "أجب بصيغة JSON فقط: {type, severity, confidence (0-100), reason, recommendation}. "
                    "إذا كانت الثقة أقل من 90 اقترح سؤال توضيحي واحد."
                )

                with st.spinner("جاري تحليل AI..."):
                    try:
                        response = client.chat.completions.create(
                            model=GPT_MODEL,
                            messages=[
                                {"role":"system","content":system_prompt},
                                {"role":"user","content":user_msg}
                            ],
                            max_tokens=700,
                            temperature=0.2
                        )
                        ai_text = response.choices[0].message.content.strip()
                    except Exception as e:
                        ai_text = f"Error: {e}"
                        st.error("حدث خطأ في الاتصال بـ OpenAI — تأكد من المفتاح والاتصال.")
                
                # save row into Excel training dataset
                row = {
                    "id": uuid.uuid4().hex,
                    "defect_name": defect_name,
                    "description": description,
                    "severity": severity,
                    "assembly_done": assembly_done,
                    "package_ok": package_ok,
                    "steps_ok": steps_ok,
                    "img_good": img_good_path,
                    "img_defect": img_defect_path,
                    "ai_guess": ai_text,
                    "user_correction": "",
                    "correction_reason": "",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                save_training_data_row(row)
                st.success("✅ تم حفظ حالة التدريب وتحليل AI. يمكنك الآن تصحيح النتيجة في قسم 'حوار التدريب'.")

    st.markdown("---")
    st.header("🗣️ حوار التدريب — صحح نتائج AI (human-in-the-loop)")

    df = load_training_data()
    pending = df[df["user_correction"].isna() | (df["user_correction"] == "")]
    st.write(f"إجمالي الحالات: {len(df)} — حالات بحاجة لتصحيح: {len(pending)}")

    mode = st.radio("اختر حالات للعرض:", ["الحالات التي تحتاج تصحيح", "كل الحالات"], index=0)
    show_df = pending.reset_index(drop=True) if mode.startswith("الحالات") else df.reset_index(drop=True)

    if not show_df.empty:
        # show table without image paths (so not to clutter)
        display_df = show_df.copy()
        if "img_good" in display_df.columns:
            display_df = display_df.drop(columns=[c for c in ["img_good","img_defect"] if c in display_df.columns])
        st.dataframe(display_df)
        sel_idx = st.number_input("اختر رقم السطر للتصحيح (index):", min_value=0, max_value=max(0,len(show_df)-1), value=0, step=1)
        sel_row = show_df.loc[sel_idx]

        st.markdown("### الحالة المحددة")
        st.write(f"**اسم العيب:** {sel_row.get('defect_name','')}")
        st.write(f"**الوصف:** {sel_row.get('description','')}")
        st.markdown("**تحليل AI السابق (JSON أو نص):**")
        st.code(sel_row.get("ai_guess",""))

        # show images from saved paths
        colA, colB = st.columns(2)
        with colA:
            img_good_p = sel_row.get("img_good","")
            if img_good_p and os.path.exists(img_good_p):
                st.image(Image.open(img_good_p), caption="الجزء السليم", use_column_width=True)
            else:
                st.info("لا توجد صورة سليمة محفوظة لهذه الحالة.")
        with colB:
            img_def_p = sel_row.get("img_defect","")
            if img_def_p and os.path.exists(img_def_p):
                st.image(Image.open(img_def_p), caption="الجزء المعيب", use_column_width=True)
            else:
                st.info("لا توجد صورة معيبة محفوظة لهذه الحالة.")

        st.markdown("#### أدخل تصحيحك كمدرّب")
        user_corr = st.text_input("التصنيف الصحيح (مثال: عيب تجميع)", key="corr_input")
        corr_reason = st.text_area("سبب التصحيح (مختصر):", key="corr_reason_input")

        if st.button("💾 حفظ التصحيح والتعليم"):
            # update training excel
            real_df = load_training_data()
            # locate by unique id if present else fallback by matching description+ai_guess timestamp
            uid = sel_row.get("id", None)
            if uid:
                idxs = real_df[real_df["id"] == uid].index.tolist()
                if idxs:
                    ix = idxs[0]
                else:
                    ix = None
            else:
                # fallback
                matches = real_df[(real_df["description"]==sel_row.get("description")) & (real_df["ai_guess"]==sel_row.get("ai_guess"))]
                ix = matches.index[0] if not matches.empty else None

            if ix is None:
                st.error("لم أتمكن من تحديد السطر بدقة للحفظ. حاول من جديد.")
            else:
                real_df.at[ix,"user_correction"] = user_corr
                real_df.at[ix,"correction_reason"] = corr_reason
                real_df.at[ix,"timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                real_df.to_excel(TRAINING_EXCEL, index=False)
                st.success("✅ تم حفظ التصحيح. سيتم استخدام هذه الحالة كمثال لتحسين ردود AI لاحقًا.")
    else:
        st.info("لا توجد حالات للعرض حالياً.")

with right:
    st.header("📦 إدارة البيانات والنتائج")
    # Results summary
    if os.path.exists(RESULTS_EXCEL):
        res_df = pd.read_excel(RESULTS_EXCEL)
    else:
        res_df = pd.DataFrame(columns=["Image Name","Result","Time"])
    st.write("إجمالي نتائج سريعة محفوظة:", len(res_df))
    if not res_df.empty:
        st.dataframe(res_df)
        buf = BytesIO()
        res_df.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button("📥 تحميل results.xlsx", data=buf.getvalue(), file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    # Training data management
    tdf = load_training_data()
    st.write("سجل التدريب (حالات):", len(tdf))
    if not tdf.empty:
        display_tdf = tdf.drop(columns=[c for c in ["img_good","img_defect"] if c in tdf.columns])
        st.dataframe(display_tdf)
        # allow download of training data
        buf2 = BytesIO()
        tdf.to_excel(buf2, index=False)
        buf2.seek(0)
        st.download_button("📥 تحميل training_data.xlsx", data=buf2.getvalue(), file_name="training_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("لم يتم إضافة حالات تدريب بعد.")

    st.markdown("---")
    if st.button("🔄 إعادة تحميل الصفحة / تحديث"):
        st.experimental_rerun()

st.markdown("---")
st.markdown("✨ Designed by Mohamed Ashraf — AI Trainer")
