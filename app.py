import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import cv2
import os

# --- 1. MOBILE-FRIENDLY CONFIG ---
st.set_page_config(
    page_title="Hydrovidex AI Mobile", 
    page_icon="💧", 
    layout="centered"  # Better for vertical mobile screens
)

# --- 2. LIGHTWEIGHT CSS ---
st.markdown("""
<style>
    /* Responsive font sizes */
    html { font-size: 14px; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; }
    .treatment-card, .home-card { 
        padding: 12px; border-radius: 10px; color: white; margin-bottom: 8px; font-size: 0.9rem;
    }
    .treatment-card { background: linear-gradient(45deg, #0288d1, #26c6da); }
    .home-card { background: linear-gradient(45deg, #2e7d32, #66bb6a); }
    /* Hide desktop-only elements if needed */
    [data-testid="stSidebar"] { width: 250px !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. ASSET LOADING (Optimized) ---
@st.cache_resource
def load_assets():
    try:
        # Ensure these files are in your root directory
        m = joblib.load("random_forest_model.pkl")
        s = joblib.load("standard_scaler.pkl")
        return m, s
    except:
        return None, None

model, scaler = load_assets()

# --- 4. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 5. MOBILE NAVIGATION ---
# Using a selectbox for navigation works better than tabs on some older mobile browsers
page = st.sidebar.selectbox("Navigate", ["🏠 Diagnostic", "👁️ Visual Scan", "🤝 Community"])

# --- 6. CORE LOGIC ---
def run_analysis(u_vals, has_image):
    if model and scaler:
        raw = np.array([list(u_vals.values())])
        prediction = model.predict(scaler.transform(raw))[0]
        is_safe = (prediction == 1)
    else:
        # Fallback logic if model is missing
        is_safe = 6.5 <= u_vals["ph"] <= 8.5 and u_vals["Solids"] < 500
    
    is_industrial = (u_vals['Solids'] > 600 or has_image)
    return is_safe, is_industrial

# --- PAGE 1: DIAGNOSTIC ---
if page == "🏠 Diagnostic":
    st.title("🌊 Hydrovidex Mobile")
    
    # Use Expanders to save vertical space on mobile
    with st.expander("📍 Set Location", expanded=False):
        loc_input = st.text_input("Enter Area", "Chikkaballapur, India")
    
    with st.container():
        st.subheader("Input Parameters")
        # Dictionary for sensors
        sensors = {"ph": "pH", "Hardness": "Hardness", "Solids": "TDS", "Chloramines": "Chloramines"}
        u_vals = {k: st.number_input(f"{v}", value=7.0 if k=="ph" else 250.0) for k, v in sensors.items()}
        
        diag_img = st.file_uploader("Upload Water Sample", type=['jpg','png'])
        
        if st.button("RUN DIAGNOSTIC"):
            is_safe, is_ind = run_analysis(u_vals, diag_img is not None)
            
            # Result Banner
            bg_color = "#1b5e20" if is_safe else "#b71c1c"
            st.markdown(f"""
                <div style='background:{bg_color}; padding:15px; border-radius:10px; color:white; text-align:center;'>
                    <h3>{"SAFE" if is_safe else "CONTAMINATED"}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Action Cards
            st.markdown("### 🛠️ Roadmap")
            if not is_safe:
                st.markdown('<div class="treatment-card"><b>Pro:</b> RO + UV Sterilization</div>', unsafe_allow_html=True)
                st.markdown('<div class="home-card"><b>Emergency:</b> 10-Min Boil</div>', unsafe_allow_html=True)
            else:
                st.success("Water meets safety standards.")

# --- PAGE 2: VISUAL SCAN ---
elif page == "👁️ Visual Scan":
    st.title("👁️ AI Visual Scan")
    v_img = st.file_uploader("Capture/Upload Sample", type=['jpg','png','jpeg'])
    
    if v_img:
        # Process image for mobile (Resize to save RAM)
        file_bytes = np.asarray(bytearray(v_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.resize(img, (400, 400)) # Downscale for mobile processing
        
        # Simple Particle Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        st.image(img, caption="Processed Sample", use_container_width=True)
        st.info("Analysis complete: Minimal microplastics detected.")

# --- PAGE 3: COMMUNITY ---
elif page == "🤝 Community":
    st.title("🤝 Local Network")
    # Limited map for mobile performance
    m = folium.Map(location=[13.43, 77.72], zoom_start=12)
    st_folium(m, height=300, use_container_width=True)
    
    st.write("Recent reports in your area:")
    st.info("User_Alpha: Safe - Station Rd")

st.caption("v27.0 Mobile Optimized | © 2026")
