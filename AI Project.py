import streamlit as st
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

RESULTS_PATH = "results.xlsx"

# ----------------- تحميل النتائج -----------------
def load_results(path=RESULTS_PATH):
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)
            # إضافة الأعمدة الجديدة لو مش موجودة
            for col in ["Code", "Supplier"]:
                if col not in df.columns:
                    df[col] = ""
            return df
        except Exception:
            return pd.DataFrame(columns=["Image Name", "Result", "Time", "Code", "Supplier"])
    else:
        return pd.DataFrame(columns=["Image Name", "Result", "Time", "Code", "Supplier"])

# ----------------- حفظ النتائج -----------------
def save_results(df, path=RESULTS_PATH):
    df.to_excel(path, index=False)

# ----------------- حذف النتائج -----------------
def clear_results(path=RESULTS_PATH):
    if os.path.exists(path):
        os.remove(path)

# ----------------- تحويل لملف Excel -----------------
def results_to_excel_bytes(df):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

# ----------------- واجهة إدارة النتائج -----------------
st.set_page_config(page_title="📊 Results Management", page_icon="📋", layout="wide")

st.title("📋 Results Management")
st.markdown("أدخل بيانات إضافية (الكود + المورد)، وشاهد أو احذف نتائج التصنيف.")

# تحميل البيانات
df = load_results()

# نموذج إدخال البيانات الجديدة
st.subheader("➕ إضافة بيانات جديدة يدوياً")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    img_name = st.text_input("🖼️ اسم الصورة", placeholder="example.jpg")
with col2:
    result = st.selectbox("📌 النتيجة", ["", "سليمة", "عيب مورد", "عيب تجميع"])
with col3:
    code = st.text_input("🧾 الكود", placeholder="EX12345")
with col4:
    supplier = st.text_input("🏢 المورد", placeholder="Company A")
with col5:
    st.write("")
    add_btn = st.button("💾 حفظ")

if add_btn:
    if img_name and result:
        new_row = {
            "Image Name": img_name,
            "Result": result,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Code": code,
            "Supplier": supplier
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_results(df)
        st.success("✅ تم إضافة البيانات وحفظها بنجاح.")
        st.experimental_rerun()
    else:
        st.warning("⚠️ من فضلك أدخل على الأقل اسم الصورة والنتيجة.")

st.markdown("---")

# عرض النتائج الحالية
st.subheader("📊 النتائج المسجلة")

if df.empty:
    st.warning("⚠️ لا توجد نتائج حالياً.")
else:
    st.write(f"إجمالي النتائج: **{len(df)}**")
    st.dataframe(df, use_container_width=True)

    # تحديد صف للحذف
    selected_index = st.number_input("اكتب رقم الصف الذي تريد حذفه (ابدأ من 0):", min_value=0, max_value=len(df)-1 if len(df)>0 else 0, step=1)

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("🗑️ حذف الصف المحدد"):
            if len(df) > 0:
                df = df.drop(df.index[selected_index])
                df.reset_index(drop=True, inplace=True)
                save_results(df)
                st.success(f"تم حذف الصف رقم {selected_index} بنجاح ✅")
                st.experimental_rerun()
            else:
                st.warning("⚠️ لا توجد نتائج لحذفها.")

    with col2:
        if st.button("❌ حذف كل النتائج"):
            clear_results()
            st.success("تم حذف جميع النتائج ✅")
            st.experimental_rerun()

    with col3:
        excel_bytes = results_to_excel_bytes(df)
        st.download_button(
            label="⬇️ تحميل النتائج (Excel)",
            data=excel_bytes,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
st.markdown('<div style="text-align:center; font-weight:bold;">✨ Designed by Mohamed Ashraf ✨</div>', unsafe_allow_html=True)
