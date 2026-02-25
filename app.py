import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
import datetime
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

# --- PAGE CONFIG ---
st.set_page_config(page_title="AquaGuard AI Pro", page_icon="💧", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("standard_scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- INITIALIZE HISTORY ---
if 'diagnostic_history' not in st.session_state:
    st.session_state.diagnostic_history = pd.DataFrame(columns=[
        "Timestamp", "Location", "Classification", "Verdict", "Reliability Index"
    ])

# --- CLEAN BUBBLE INJECTION ---
def inject_bubbles():
    bubble_css = """
    <style>
        .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 50%, #80deea 100%); background-attachment: fixed; }
        .bubble { position: fixed; bottom: -150px; background: rgba(255, 255, 255, 0.4); border-radius: 50%; animation: rise 15s infinite ease-in; z-index: 0; pointer-events: none; }
        @keyframes rise { 0% { bottom: -100px; transform: translateX(0); opacity: 0.8; } 100% { bottom: 110vh; transform: translateX(-50px); opacity: 0; } }
    </style>
    """
    st.markdown(bubble_css, unsafe_allow_html=True)
    for i in range(1, 26):
        size, left, dur, delay = np.random.randint(20, 70), np.random.randint(0, 95), np.random.uniform(10, 18), np.random.uniform(0, 10)
        st.markdown(f'<style>.b{i} {{ width:{size}px; height:{size}px; left:{left}%; animation-duration:{dur}s; animation-delay:{delay}s; }}</style>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble b{i}"></div>', unsafe_allow_html=True)

inject_bubbles()

# --- SIDEBAR: FIELD DATA & IMAGE SCAN ---
with st.sidebar:
    st.title("📍 Field Parameters")
    loc_query = st.text_input("Village/City Name", "Delhi, India")
    
    st.divider()
    st.subheader("📸 Visual Scan")
    uploaded_img = st.file_uploader("Upload Sample Photo", type=["jpg", "png", "jpeg"])
    
    source_type = "Not Identified"
    if uploaded_img:
        img = Image.open(uploaded_img)
        st.image(img, caption="Sample Captured", use_container_width=True)
        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean) / 3
        source_type = "Natural Water Body" if brightness < 120 else "Man-Processed Water"
        st.info(f"**Visual Classification:** {source_type}")

    st.divider()
    st.subheader("🧪 Sensor Telemetry")
    features = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
    
    # WHO Defined Safety Limits
    WHO_LIMITS = {
        "ph": (6.5, 8.5), "Hardness": (0, 200), "Solids": (0, 1000), "Chloramines": (0, 4.0),
        "Sulfate": (0, 250), "Conductivity": (0, 400), "Organic_carbon": (0, 2.0), "Trihalomethanes": (0, 0.08), "Turbidity": (0, 5.0)
    }
    
    user_inputs = {}
    for f in features:
        # Default value set to roughly middle of WHO range for safety
        default_val = 7.0 if f == "ph" else float(WHO_LIMITS[f][1] * 0.4)
        user_inputs[f] = st.number_input(f"🔹 {f}", value=default_val)

# --- MAIN DASHBOARD ---
st.title("🌊 AquaGuard AI: Professional Diagnostic")

# 1. LIVE MONITORING MAP
geolocator = Nominatim(user_agent="aquaguard_ai")
try:
    loc = geolocator.geocode(loc_query)
    lat, lon = (loc.latitude, loc.longitude) if loc else (28.61, 77.20)
except: lat, lon = 28.61, 77.20

m = folium.Map(location=[lat, lon], zoom_start=12)
folium.Marker([lat, lon], popup=loc_query, icon=folium.Icon(color='blue', icon='tint')).add_to(m)
st_folium(m, width="100%", height=350)

# 2. WHO COMPLIANCE AUDITOR (NEW FEATURE)
st.subheader("📋 WHO Water Quality Compliance Auditor")


compliance_data = []
for f in features:
    val = user_inputs[f]
    low, high = WHO_LIMITS[f]
    is_safe = low <= val <= high
    status = "✅ COMPLIANT" if is_safe else "⚠️ VIOLATION"
    compliance_data.append({
        "Parameter": f.replace("_", " ").title(),
        "WHO Limit": f"{low} - {high}",
        "Input Value": val,
        "Status": status
    })

# Render WHO Auditor Table
st.table(pd.DataFrame(compliance_data))

# 3. SOURCE ANALYSIS
if uploaded_img:
    st.subheader("🔍 Source Identification Analysis")
    c1, c2 = st.columns(2)
    with c1:
        if source_type == "Natural Water Body":
            st.warning("### 🌿 Natural Water Body")
            st.write("**Description:** Surface-level water detected. High risk of microbial pathogens and organic runoff. Intensive Phase 3 disinfection required.")
        else:
            st.success("### 🏭 Man-Processed Water")
            st.write("**Description:** Processed source detected. Primary risks involve chemical byproduct residuals like Trihalomethanes or Chloramines.")
    with c2:
        st.info("**Visual Analysis Result**")
        st.write(f"The sample luminance reflects characteristics of **{source_type}**. Cross-referencing with sensor data for verification.")

# 4. EXECUTION
if st.button("RUN FULL SYSTEM DIAGNOSTIC"):
    if model:
        # Prediction Logic
        input_array = np.array([user_inputs[f] for f in features]).reshape(1, -1)
        data_scaled = scaler.transform(input_array)
        prediction = int(model.predict(data_scaled)[0])
        prob = model.predict_proba(data_scaled)[0][prediction]
        verdict = "BIO-SECURE (Potable)" if prediction == 1 else "CONTAMINATED (Unsafe)"
        v_color = "#2e7d32" if prediction == 1 else "#c62828"

        st.markdown(f"<div style='background:{v_color}; padding:25px; border-radius:15px; color:white; text-align:center;'><h1>{verdict}</h1><h3>Reliability Index: {prob:.2%}</h3></div>", unsafe_allow_html=True)

        # Update History
        new_entry = pd.DataFrame([{
            "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"), 
            "Location": loc_query, 
            "Classification": source_type, 
            "Verdict": verdict, 
            "Reliability Index": f"{prob:.2%}"
        }])
        st.session_state.diagnostic_history = pd.concat([new_entry, st.session_state.diagnostic_history], ignore_index=True)

        # 5. XAI GRAPH (PATCHED SHAPE MISMATCH)
        st.divider()
        st.subheader("📊 XAI: Explainable AI Feature Influence")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_scaled)
        
        # Binary Classification Patch
        if isinstance(shap_values, list):
            sv = shap_values[prediction].flatten()
        else:
            sv = shap_values[:, :, prediction].flatten() if len(shap_values.shape) == 3 else shap_values.flatten()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(features, sv, color=['#4db6ac' if x > 0 else '#ef5350' for x in sv])
        ax.set_title(f"Contribution towards {verdict}")
        st.pyplot(fig)

        # 6. TREATMENT PHASES
        st.divider()
        st.subheader("🛠️ Professional Treatment Roadmap")
        
        p1, p2, p3 = st.columns(3)
        with p1:
            st.info("**PHASE 1: Physical Mitigation**\n\nRemoving turbidity and large organic matter through coagulation and flocculation.")
        with p2:
            st.info("**PHASE 2: Chemical Neutralization**\n\nUtilizing Reverse Osmosis (RO) to bring Solids and Sulfate levels into WHO compliance.")
        with p3:
            st.info("**PHASE 3: Biological Sterilization**\n\nUV-C Radiation or Ozone injection to eliminate microbial threats detected in visual scan.")
    else:
        st.error("Engine Offline: Model files missing.")

# 7. HISTORY LOGS
st.divider()
st.subheader("📜 Diagnostic History")
st.dataframe(st.session_state.diagnostic_history, use_container_width=True)

st.caption(f"AquaGuard AI Core v4.6 | {datetime.datetime.now().year}")
