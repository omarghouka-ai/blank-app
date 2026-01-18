# 🧠 AI Stroke Risk Prediction App

An interactive **Streamlit-based web application** that uses **machine learning (CatBoost)** to predict the probability of stroke based on patient medical, lifestyle, and demographic data.

The app supports **English and Arabic**, includes **RTL layout handling**, and provides **visual risk analysis** using Plotly charts.

---

## 🚀 Features

- 🌍 **Bilingual Interface** (English / العربية)
- 🧠 **AI-powered stroke risk prediction**
- 📊 **Interactive visualizations**
  - Probability gauge
  - Patient vs population comparison
- 🩺 Handles **missing BMI intelligently**
- 🧭 Right-to-left (RTL) support for Arabic
- ⚡ Fast inference using a pre-trained CatBoost model

---

## 🖥️ Demo Screens

- Patient data entry form
- Stroke probability gauge
- Risk classification (High / Low)
- Medical warnings & insights

---

## 🧪 Model Overview

- **Algorithm:** CatBoostClassifier  
- **Task:** Binary classification (Stroke / No Stroke)  
- **Output:** Probability score (0–100%)  
- **Threshold:** `0.66` (≥ 66% → High Risk)

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/stroke-risk-ai.git
cd stroke-risk-ai
```

### 2️⃣ Create a virtual environment (recommended
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run the App
```bash
streamlit run app.py
```

# Make sure the trained model file exists: stroke_model.cbm


## 📜 requirements.txt
```bash
streamlit
pandas
numpy
catboost
plotly
```
