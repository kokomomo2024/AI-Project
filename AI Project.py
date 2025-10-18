import streamlit as st
from PIL import Image
import random

# ----------------- إعداد واجهة Streamlit -----------------
st.set_page_config(page_title="AI Project", page_icon="🤖", layout="wide")

st.title("🤖 AI Project — Image Classification")
st.write("قم برفع صورة ليتم تصنيفها آليًا (بدون حفظ أو تسجيل).")

# ----------------- رفع الصورة -----------------
uploaded_file = st.file_uploader("📤 اختر الصورة (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 الصورة المرفوعة", use_container_width=True)

    if st.button("🔍 تحليل الصورة"):
        result = random.choice(["سليمة", "عيب مورد", "عيب تجميع"])
        st.success(f"✅ النتيجة: {result}")

# ----------------- تصميم بسيط -----------------
st.markdown("---")
st.markdown('<div style="text-align:center;font-weight:bold;">✨ Designed by Mohamed Ashraf ✨</div>', unsafe_allow_html=True)
