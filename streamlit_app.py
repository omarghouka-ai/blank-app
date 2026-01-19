import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import plotly.graph_objects as go
import os

# --- ADDED FOR IMAGE CLASSIFICATION ---
import torch
from PIL import Image
import timm
import torchvision.transforms as transforms

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Stroke Risk AI / الذكاء الاصطناعي لتوقع السكتة الدماغية",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. TRANSLATION DICTIONARY ---
T = {
    "en": {
        "title": "🧠 AI Stroke Risk Prediction",
        "subtitle": "Enter patient medical details below to estimate stroke probability.",
        "personal_info": "Personal Information",
        "medical_history": "Medical History",
        "vitals_lifestyle": "Vitals & Lifestyle",
        "gender": "Gender",
        "age": "Age",
        "residence": "Residence Type",
        "married": "Ever Married?",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease",
        "work": "Work Type",
        "glucose": "Avg Glucose Level (mg/dL)",
        "bmi_check": "Patient BMI is known",
        "bmi": "Body Mass Index (BMI)",
        "bmi_placeholder": "BMI assumed unknown (auto-handled)",
        "smoking": "Smoking Status",
        "predict_btn": "Analyze Risk",
        "result_header": "Risk Analysis Result",
        "prob_label": "Stroke Probability",
        "high_risk": "High Risk",
        "low_risk": "Low Risk",
        "factors_header": "Risk Factor Analysis",
        "chart_title": "Patient Vitals vs Population Average",
        "warning_bp": "⚠️ Hypertension is a significant risk factor.",
        "warning_heart": "⚠️ History of Heart Disease increases risk.",
        "info_age": "ℹ️ Age is a non-modifiable risk factor.",
        "male": "Male", "female": "Female",
        "urban": "Urban", "rural": "Rural",
        "yes": "Yes", "no": "No",
        "private": "Private", "self_emp": "Self-employed", "govt": "Govt_job", "children": "children", "never": "Never_worked",
        "formerly": "formerly smoked", "never_sm": "never smoked", "smokes": "smokes", "unknown": "Unknown",
        "pat_glucose": "Patient Glucose", "avg_glucose": "Avg Pop. Glucose",
        "pat_bmi": "Patient BMI", "avg_bmi": "Avg Pop. BMI",
        "loading_err": "Model file not found. Please verify 'stroke_model.cbm' is in the folder.",
        "dir": "ltr",
        "align": "left"
    },
    "ar": {
        "title": "🧠 الذكاء الاصطناعي لتوقع السكتة الدماغية",
        "subtitle": "أدخل البيانات الطبية للمريض أدناه لتقدير احتمالية الإصابة.",
        "personal_info": "البيانات الشخصية",
        "medical_history": "التاريخ الطبي",
        "vitals_lifestyle": "المؤشرات الحيوية ونمط الحياة",
        "gender": "الجنس",
        "age": "العمر",
        "residence": "نوع الإقامة",
        "married": "هل سبق الزواج؟",
        "hypertension": "ارتفاع ضغط الدم",
        "heart_disease": "أمراض القلب",
        "work": "نوع العمل",
        "glucose": "متوسط مستوى الجلوكوز",
        "bmi_check": "مؤشر كتلة الجسم معروف",
        "bmi": "مؤشر كتلة الجسم (BMI)",
        "bmi_placeholder": "سيتم التعامل مع الوزن كغير معروف",
        "smoking": "حالة التدخين",
        "predict_btn": "تحليل المخاطر",
        "result_header": "نتيجة التحليل",
        "prob_label": "احتمالية الإصابة",
        "high_risk": "خطر مرتفع",
        "low_risk": "خطر منخفض",
        "factors_header": "تحليل عوامل الخطر",
        "chart_title": "مقارنة المريض بمتوسط السكان",
        "warning_bp": "⚠️ ارتفاع ضغط الدم هو عامل خطر كبير.",
        "warning_heart": "⚠️ تاريخ أمراض القلب يزيد من المخاطر.",
        "info_age": "ℹ️ العمر عامل خطر لا يمكن تغييره.",
        "male": "ذكر", "female": "أنثى",
        "urban": "حضر", "rural": "ريف",
        "yes": "نعم", "no": "لا",
        "private": "قطاع خاص", "self_emp": "عمل حر", "govt": "وظيفة حكومية", "children": "أطفال", "never": "لم يعمل أبداً",
        "formerly": "مدخن سابق", "never_sm": "غير مدخن", "smokes": "مدخن حالي", "unknown": "غير معروف",
        "pat_glucose": "جلوكوز المريض", "avg_glucose": "متوسط الجلوكوز العام",
        "pat_bmi": "كتلة جسم المريض", "avg_bmi": "متوسط كتلة الجسم",
        "loading_err": "ملف النموذج غير موجود. يرجى التأكد من وجود 'stroke_model.cbm'",
        "dir": "rtl",
        "align": "right"
    }
}

# --- ADDED TRANSLATIONS FOR IMAGE CLASSIFICATION + MODE SELECTOR ---
T["en"].update({
    "mode_label": "Choose input method",
    "mode_clinical": "Enter medical data",
    "mode_image": "Upload image",
    "img_upload": "Upload an image",
    "img_hint": "Upload an image, then press Analyze Risk.",
    "img_model_err": "Image model could not be loaded. Please verify 'vit_fold5.pth' and required libraries are installed.",
    "img_no_image": "Please upload an image first.",
    "img_result": "Image Classification Result",
    "img_topk": "Top Predictions",
})
T["ar"].update({
    "mode_label": "اختر طريقة الإدخال",
    "mode_clinical": "إدخال البيانات الطبية",
    "mode_image": "رفع صورة",
    "img_upload": "ارفع صورة",
    "img_hint": "ارفع صورة ثم اضغط تحليل المخاطر.",
    "img_model_err": "تعذر تحميل نموذج الصور. تأكد من وجود 'vit_fold5.pth' وتثبيت المكتبات المطلوبة.",
    "img_no_image": "يرجى رفع صورة أولاً.",
    "img_result": "نتيجة تصنيف الصورة",
    "img_topk": "أفضل التوقعات",
})

# --- 3. LOAD STROKE MODEL ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    try:
        model.load_model("stroke_model.cbm")
        return model
    except Exception:
        return None

model = load_model()

# ==========================================================
# IMAGE MODEL (ViT) - uses vit_fold5.pth
# ==========================================================
VIT_WEIGHTS_PATH = "vit_fold5.pth"

# IMPORTANT: set this to the exact ViT variant you trained with.
# Examples: "vit_base_patch16_224", "vit_small_patch16_224", "vit_large_patch16_224"
VIT_ARCH = "vit_base_patch16_224"

# OPTIONAL (recommended): set your class labels in the exact training order.
# If left empty, the app will show "Class 0", "Class 1", ...
CLASS_NAMES = [
    # "label_0", "label_1", ...
]

@st.cache_resource
def load_vit_classifier():
    try:
        if not os.path.exists(VIT_WEIGHTS_PATH):
            raise FileNotFoundError(f"Not found: {VIT_WEIGHTS_PATH} (cwd={os.getcwd()})")

        ckpt = torch.load(VIT_WEIGHTS_PATH, map_location="cpu")

        # Case A: whole model object saved (torch.save(model, ...))
        if hasattr(ckpt, "state_dict"):
            state = ckpt.state_dict()
        # Case B: dict checkpoint
        elif isinstance(ckpt, dict):
            state = (ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt.get("model") or ckpt)
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

        # Clean common prefixes (module., model.)
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            cleaned[nk] = v
        state = cleaned

        # Infer num_classes from head/classifier weights if present
        inferred_num_classes = None
        for k in ["head.weight", "classifier.weight", "fc.weight"]:
            if k in state and hasattr(state[k], "shape"):
                inferred_num_classes = int(state[k].shape[0])
                break

        num_classes = len(CLASS_NAMES) if CLASS_NAMES else (inferred_num_classes or 2)
        model_img = timm.create_model(VIT_ARCH, pretrained=False, num_classes=num_classes)

        # Remove mismatched keys to prevent size mismatch errors
        model_state = model_img.state_dict()
        to_delete = []
        for k, v in state.items():
            if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                if tuple(v.shape) != tuple(model_state[k].shape):
                    to_delete.append(k)
        for k in to_delete:
            del state[k]

        model_img.load_state_dict(state, strict=False)
        model_img.eval()
        return model_img, None
    except Exception as e:
        return None, str(e)

def vit_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def predict_image(model_img, pil_img: Image.Image, topk: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_img = model_img.to(device)

    img = pil_img.convert("RGB")
    x = vit_preprocess()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_img(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=k)

    results = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        label = CLASS_NAMES[i] if CLASS_NAMES and i < len(CLASS_NAMES) else f"Class {i}"
        results.append((label, v))
    return results

# --- 4. LANGUAGE SELECTOR & CSS INJECTION ---
col_logo, col_lang = st.columns([8, 2])
with col_lang:
    lang_choice = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True, label_visibility="collapsed")
    lang = "en" if lang_choice == "English" else "ar"

# --- GLOBAL STYLES (Applies to both AR and EN) ---
st.markdown(
    """
    <style>
    /* Add padding to all checkboxes */
    .stCheckbox {
        padding-left: 8px;
        padding-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- ARABIC SPECIFIC STYLES ---
if lang == "ar":
    st.markdown(
        """
        <style>
        .stApp { direction: rtl; text-align: right; }
        .stSelectbox, .stNumberInput, .stRadio, .stCheckbox, .stMetric, p, h1, h2, h3, .stAlert { text-align: right; }
        div[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
        .st-bl { padding-right: 8px; }
        .st-en { padding-right: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
if lang == "ar":
    st.markdown(
        """
        <style>
        .stApp { direction: rtl; text-align: right; }
        .stSelectbox, .stNumberInput, .stRadio, .stCheckbox, .stMetric, p, h1, h2, h3, .stAlert { text-align: right; }
        div[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Title
st.title(T[lang]["title"])
st.markdown(f"*{T[lang]['subtitle']}*")
st.markdown("---")

# Helper dictionaries
gender_map = {T[lang]["male"]: "Male", T[lang]["female"]: "Female"}
yes_no_map = {T[lang]["yes"]: "Yes", T[lang]["no"]: "No"}
residence_map = {T[lang]["urban"]: "Urban", T[lang]["rural"]: "Rural"}
work_map = {
    T[lang]["private"]: "Private", T[lang]["self_emp"]: "Self-employed",
    T[lang]["govt"]: "Govt_job", T[lang]["children"]: "children", T[lang]["never"]: "Never_worked"
}
smoking_map = {
    T[lang]["formerly"]: "formerly smoked", T[lang]["never_sm"]: "never smoked",
    T[lang]["smokes"]: "smokes", T[lang]["unknown"]: "Unknown"
}

# --- INPUT MODE SELECTOR (before medical form & before Analyze button) ---
mode = st.radio(
    T[lang]["mode_label"],
    [T[lang]["mode_clinical"], T[lang]["mode_image"]],
    horizontal=True
)

img_file = None
pil_img = None

if mode == T[lang]["mode_image"]:
    st.info(T[lang]["img_hint"])
    col_up, col_prev = st.columns([2, 1])
    with col_up:
        img_file = st.file_uploader(
            T[lang]["img_upload"],
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False
        )
    with col_prev:
        if img_file is not None:
            pil_img = Image.open(img_file)
            st.image(pil_img, use_container_width=True)

st.markdown("---")

# --- INPUT FORM (only in clinical mode) ---
if mode == T[lang]["mode_clinical"]:
    with st.container():
        # Section 1: Personal Info
        st.subheader(f"👤 {T[lang]['personal_info']}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender_ui = st.selectbox(T[lang]["gender"], list(gender_map.keys()))
        with c2:
            age = st.number_input(T[lang]["age"], 0, 120, 50)
        with c3:
            residence_ui = st.selectbox(T[lang]["residence"], list(residence_map.keys()))
        with c4:
            married_ui = st.selectbox(T[lang]["married"], list(yes_no_map.keys()))

        st.markdown("---")

        # Section 2: Vitals & Medical
        c_med, c_vit = st.columns([1, 2])

        with c_med:
            st.subheader(f"❤️ {T[lang]['medical_history']}")
            st.markdown("<br>", unsafe_allow_html=True)
            hypertension = st.checkbox(T[lang]["hypertension"])
            st.markdown("<br>", unsafe_allow_html=True)
            heart_disease = st.checkbox(T[lang]["heart_disease"])

        with c_vit:
            st.subheader(f"📊 {T[lang]['vitals_lifestyle']}")

            ra_c1, ra_c2 = st.columns(2)
            with ra_c1:
                work_ui = st.selectbox(T[lang]["work"], list(work_map.keys()))
            with ra_c2:
                smoking_ui = st.selectbox(T[lang]["smoking"], list(smoking_map.keys()))

            bmi_known = st.checkbox(T[lang]["bmi_check"], value=True)

            rc_c1, rc_c2 = st.columns(2)
            with rc_c1:
                if bmi_known:
                    bmi = st.number_input(T[lang]["bmi"], 10.0, 100.0, 28.0)
                    bmi_missing_val = 0
                else:
                    st.text_input(T[lang]["bmi"], value=T[lang]["bmi_placeholder"], disabled=True)
                    bmi = 0.0
                    bmi_missing_val = 1
            with rc_c2:
                avg_glucose = st.number_input(T[lang]["glucose"], 50.0, 300.0, 100.0)

# --- PREDICTION PROCESSING ---
st.markdown("<br><br>", unsafe_allow_html=True)

# Centered Button Logic (one button for both modes)
col_space1, col_btn, col_space2 = st.columns([5, 3, 5])
with col_btn:
    predict_pressed = st.button(T[lang]["predict_btn"], type="primary", use_container_width=True)

if predict_pressed:

    # =========================
    # IMAGE MODE PREDICTION
    # =========================
    if mode == T[lang]["mode_image"]:
        vit_model, vit_err = load_vit_classifier()

        if vit_model is None:
            st.error(T[lang]["img_model_err"])
            if vit_err:
                with st.expander("Details", expanded=False):
                    st.code(vit_err)
                    st.write("CWD:", os.getcwd())
                    st.write("Files:", os.listdir())
                    st.write("vit_fold5.pth exists?", os.path.exists("vit_fold5.pth"))
        elif pil_img is None:
            st.warning(T[lang]["img_no_image"])
        else:
            results = predict_image(vit_model, pil_img, topk=5)
            best_label, best_prob = results[0]

            st.markdown("---")
            col_res, col_chart = st.columns([1, 2])

            with col_res:
                st.subheader(T[lang]["img_result"])
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid #3182ce; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05);">
                    <h2 style="color: #3182ce; margin:0;">{best_label}</h2>
                    <h1 style="font-size: 50px; margin:0;">{best_prob*100:.1f}%</h1>
                    <p>{T[lang]["prob_label"]}</p>
                </div>
                """, unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=best_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3182ce"},
                        'steps': [{'range': [0, 100], 'color': "#e6fffa"}],
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_chart:
                st.subheader(T[lang]["img_topk"])
                for label, prob in results:
                    st.write(f"{label} — {prob*100:.1f}%")
                    st.progress(int(prob * 100))

    # =========================
    # CLINICAL MODE PREDICTION
    # =========================
    else:
        if model:
            gender_val = gender_map[gender_ui]
            hypertension_val = 1 if hypertension else 0
            heart_disease_val = 1 if heart_disease else 0
            ever_married_val = yes_no_map[married_ui]
            work_val = work_map[work_ui]
            residence_val = residence_map[residence_ui]
            smoking_val = smoking_map[smoking_ui]
            log_glucose_val = np.log1p(avg_glucose)

            data = {
                'gender': gender_val, 'age': age, 'hypertension': hypertension_val,
                'heart_disease': heart_disease_val, 'ever_married': ever_married_val,
                'work_type': work_val, 'Residence_type': residence_val,
                'avg_glucose_level': avg_glucose, 'bmi': bmi,
                'smoking_status': smoking_val, 'bmi_missing': bmi_missing_val,
                'log_glucose': log_glucose_val
            }

            cols = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                    'smoking_status', 'bmi_missing', 'log_glucose']

            input_df = pd.DataFrame(data, index=[0])[cols]
            prediction_prob = model.predict_proba(input_df)[0][1]

            threshold = 0.66
            is_high_risk = prediction_prob >= threshold
            risk_text = T[lang]["high_risk"] if is_high_risk else T[lang]["low_risk"]
            color = "red" if is_high_risk else "green"

            st.markdown("---")
            col_res, col_chart = st.columns([1, 2])

            with col_res:
                st.subheader(T[lang]["result_header"])
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid {color}; padding: 20px; border-radius: 10px; background-color: rgba(255,255,255,0.05);">
                    <h2 style="color: {color}; margin:0;">{risk_text}</h2>
                    <h1 style="font-size: 50px; margin:0;">{prediction_prob*100:.1f}%</h1>
                    <p>{T[lang]["prob_label"]}</p>
                </div>
                """, unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 66], 'color': "#e6fffa"},
                            {'range': [66, 100], 'color': "#fff5f5"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 66}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_chart:
                st.subheader(T[lang]["factors_header"])

                categories = [T[lang]["pat_glucose"], T[lang]["avg_glucose"], T[lang]["pat_bmi"], T[lang]["avg_bmi"]]
                values = [avg_glucose, 106.0, bmi if bmi_missing_val == 0 else 0, 28.9]
                colors = ['#3182ce', '#a0aec0', '#3182ce', '#a0aec0']
                if avg_glucose > 140: colors[0] = '#e53e3e'
                if bmi > 30: colors[2] = '#e53e3e'

                fig_bar = go.Figure(data=[go.Bar(
                    x=categories,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.1f}" for v in values],
                    textposition='auto',
                )])

                title_align = 1.0 if lang == "ar" else 0.0
                fig_bar.update_layout(
                    title={'text': T[lang]["chart_title"], 'x': title_align},
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                if hypertension_val == 1: st.warning(T[lang]["warning_bp"])
                if heart_disease_val == 1: st.warning(T[lang]["warning_heart"])
                if age > 60: st.info(T[lang]["info_age"])

        else:
            st.error(T[lang]["loading_err"])
