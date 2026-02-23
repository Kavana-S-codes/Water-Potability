import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF 
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AquaGuard AI Pro", page_icon="💧", layout="wide")

# --- INITIALIZE HISTORY STORAGE ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "Location", "Type", "Verdict", "Reliability"])

# --- LIGHT TURQUOISE DESIGN ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 50%, #80deea 100%);
        background-attachment: fixed;
    }
    .bubble {
        position: fixed;
        bottom: -150px;
        background: rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 50%;
        animation: rise 15s infinite ease-in;
        z-index: 0;
        pointer-events: none;
    }
    @keyframes rise {
        0% { bottom: -150px; transform: translateX(0); opacity: 0.8; }
        50% { transform: translateX(50px); opacity: 0.4; }
        100% { bottom: 110vh; transform: translateX(-30px); opacity: 0; }
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(15px);
        border-right: 2px solid #00acc1;
    }
    [data-testid="stSidebar"] label { color: #004d40 !important; font-weight: 900 !important; }
    h1, h2, h3 { color: #006064 !important; font-weight: 900 !important; }
    p, .stMarkdown { color: #004d40 !important; font-weight: 500; }
    .stAlert, div.potable-box, div.non-potable-box, [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(20px);
        border: 2px solid #00acc1;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00acc1 0%, #26c6da 100%) !important;
        color: #ffffff !important;
        border-radius: 50px;
        font-size: 22px !important;
        font-weight: bold !important;
        width: 100%;
        border: 2px solid white;
    }
    </style>
    <div class="bubble" style="width:50px; height:50px; left:15%; animation-duration:10s;"></div>
    <div class="bubble" style="width:70px; height:70px; left:75%; animation-duration:12s; animation-delay:5s;"></div>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📍 SITE TRACKING")
    loc_name = st.text_input("Sample Source ID", "Station-Alpha-01").strip()
    is_natural = st.toggle("Environmental Resource (River/Lake)")
    water_type = "Environmental" if is_natural else "Anthropogenic/Processed"
    
    st.divider()
    st.markdown("## 🧪 ANALYSIS PARAMETERS")
    features = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
    input_dict = {f: st.number_input(f"🔹 {f}", value=7.0 if f=='ph' else 150.0) for f in features}

# --- MAIN UI ---
st.title("🌊 AquaGuard AI: Professional Diagnostic")
st.write(f"**Tracking ID:** {loc_name} | **Matrix Type:** {water_type}")

if st.button("INITIATE ANALYTICAL SCAN"):
    # --- UNIQUE ID CHECK ---
    # Check if the loc_name already exists in the history dataframe
    if not st.session_state.history.empty and loc_name in st.session_state.history['Location'].values:
        st.error(f"❌ **Duplicate Entry Blocked:** ID '{loc_name}' has already been processed in this session. Please use a unique ID for new samples.")
    else:
        # 1. Processing
        data_raw = np.array(list(input_dict.values())).reshape(1, -1)
        data_scaled = scaler.transform(data_raw)
        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0][prediction]
        verdict_main = "BIO-SECURE (SAFE)" if prediction == 1 else "CONTAMINATED (UNSAFE)"

        # 2. History Logging
        new_entry = pd.DataFrame({
            "Timestamp": [datetime.datetime.now().strftime("%H:%M:%S")],
            "Location": [loc_name],
            "Type": [water_type],
            "Verdict": [verdict_main],
            "Reliability": [f"{prob:.2%}"]
        })
        st.session_state.history = pd.concat([new_entry, st.session_state.history], ignore_index=True)

        # 3. Results Display
        col1, col2 = st.columns(2)
        with col1:
            v_color = "#00796b" if prediction == 1 else "#c62828"
            st.markdown(f'''<div class="{"potable-box" if prediction==1 else "non-potable-box"}" style="padding:25px; border-left:10px solid {v_color};">
                <h2 style="color:{v_color}; margin-top:0;">{verdict_main}</h2>
                <p>Diagnostic Reliability Index: <b>{prob:.2%}</b></p>
                </div>''', unsafe_allow_html=True)

        with col2:
            st.subheader("🛠 Treatment Recommendation")
            if prediction == 1 and prob < 0.85:
                st.warning("⚠️ **Marginal Reliability:** Verdict is SAFE, but Index is below 85%. Secondary laboratory verification is highly recommended.")
            elif prediction == 1:
                st.success("💎 **Verified Potable:** Sample aligns with high-certainty safety thresholds.")
            
            if prediction == 0:
                if is_natural:
                    st.error("🚨 **Remediation:** Natural resource contamination detected. Biological and mineral filtration required.")
                else:
                    st.error("🚨 **Process Deviation:** Anthropogenic sample exceeds safety limits. Immediate RO-UV intervention mandated.")

        # --- XAI CHART ---
        st.divider()
        st.subheader("📊 Molecular Feature Sensitivity")
        try:
            explainer = shap.TreeExplainer(model)
            shap_v = explainer.shap_values(data_scaled)
            if isinstance(shap_v, list):
                display_vals = np.array(shap_v[prediction]).flatten()
            else:
                display_vals = np.array(shap_v).flatten() if len(shap_v.shape) <= 2 else np.array(shap_v[0, :, prediction]).flatten()

            fig, ax = plt.subplots(figsize=(10, 5))
            y_pos = np.arange(len(features))
            colors = ['#00897b' if x > 0 else '#e53935' for x in display_vals]
            ax.barh(y_pos, display_vals, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            fig.patch.set_facecolor('#ffffff00') 
            ax.set_facecolor('#ffffff')          
            ax.patch.set_alpha(0.3)              
            ax.tick_params(colors='#004d40', labelsize=10)
            for spine in ax.spines.values(): spine.set_color('#00acc1')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Visualization Engine Exception: {e}")

        # 4. Export Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"AquaGuard AI Analysis Certificate", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Sample ID: {loc_name}", ln=True)
        pdf.cell(0, 10, f"Diagnostic Verdict: {verdict_main}", ln=True)
        pdf.cell(0, 10, f"Reliability Index: {prob:.2%}", ln=True)
        pdf_output = bytes(pdf.output(dest='S'))
        st.download_button("📄 Export Analytical Report", data=pdf_output, file_name=f"Report_{loc_name}.pdf")

# --- HISTORY ---
st.divider()
st.subheader("📜 Analytical Session Logs")
if not st.session_state.history.empty:
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)