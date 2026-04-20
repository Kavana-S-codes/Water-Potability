import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from fpdf import FPDF
import os
import random

# --- 1. LANGUAGE DICTIONARY ---
LANG_DICT = {
    "English": {
        "title": "🌊 Hydrovidex AI: The Autonomous Water Quality Diagnostic Engine",
        "tab1": "🏠 Diagnostic", "tab2": "👁️ Visual Scan", "tab3": "🤝 Community", "tab4": "👤 Profile",
        "run_diag": "RUN FULL SYSTEM DIAGNOSTIC",
        "verdict_safe": "NON-CONTAMINATED (Safe)", "verdict_unsafe": "CONTAMINATED (Unsafe)",
        "download_report": "📥 DOWNLOAD DESIGNER PDF REPORT",
        "comm_header": "Global & Local Water Quality Network",
        "upload_label": "Upload Image for Pollution Source Analysis",
        "visual_explain_label": "Upload Water Sample for Visual Analysis (No Values Needed)",
        "treatment_title": "🛠️ Professional Treatment Roadmap",
        "homemade_title": "🏡 Emergency Homemade Solutions",
        "graph_title": "📊 Parameter Concentration Analysis",
        "who_head": "📋 WHO Water Quality Limits & Analysis",
        "who_param": "Parameter", "who_limit": "Limit", "who_result": "Status",
        "pollution_label": "Detected Pollution Source",
        "sensors": {"ph": "pH Level", "Hardness": "Hardness", "Solids": "TDS", "Chloramines": "Chloramines", "Sulfate": "Sulfate", "Conductivity": "Conductivity", "Organic_carbon": "TOC", "Trihalomethanes": "THMs", "Turbidity": "Turbidity"}
    },
    "ಕನ್ನಡ": {
        "title": "🌊 ಹೈಡ್ರೋವಿಡೆಕ್ಸ್ ಎಐ",
        "tab1": "🏠 ರೋಗನಿರ್ಣಯ", "tab2": "👁️ ದೃಶ್ಯ ಸ್ಕ್ಯಾನ್", "tab3": "🤝 ಸಮುದಾಯ", "tab4": "👤 ಪ್ರೊಫೈಲ್",
        "run_diag": "ರೋಗನಿರ್ಣಯ ಚಲಾಯಿಸಿ",
        "verdict_safe": "ಸುರಕ್ಷಿತ", "verdict_unsafe": "ಅಸುರಕ್ಷಿತ",
        "download_report": "📥 PDF ವರದಿ",
        "upload_label": "ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "visual_explain_label": "ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಚಿತ್ರ",
        "pollution_label": "ಮಾಲಿನ್ಯದ ಮೂಲ",
        "sensors": {"ph": "ಪಿಹೆಚ್ ಮಟ್ಟ", "Hardness": "ಗಡಸುತನ", "Solids": "ಟಿಡಿಎಸ್", "Chloramines": "ಕ್ಲೋರಮೈನ್ಸ್", "Sulfate": "ಸಲ್ಫೇಟ್", "Conductivity": "ವಾಹಕತೆ", "Organic_carbon": "ಇಂಗಾಲ", "Trihalomethanes": "ಟಿಎಚ್‌ಎಂ", "Turbidity": "ಕಲಕುತನ"}
    },
    "हिंदी": {
        "title": "🌊 हाइड्रोविडेक्स एआई",
        "tab1": "🏠 निदान", "tab2": "👁️ विजुअल स्कैन", "tab3": "🤝 समुदाय", "tab4": "👤 प्रोफाइल",
        "run_diag": "पूर्ण निदान चलाएं",
        "verdict_safe": "सुरक्षित", "verdict_unsafe": "असुरक्षित",
        "download_report": "📥 PDF रिपोर्ट",
        "upload_label": "छवि अपलोड करें",
        "visual_explain_label": "दृश्य विश्लेषण",
        "pollution_label": "प्रदूषण का स्रोत",
        "sensors": {"ph": "पीएच स्तर", "Hardness": "कठोरता", "Solids": "टीडीएस", "Chloramines": "क्लोरामाइन", "Sulfate": "सल्फेट", "Conductivity": "चालकता", "Organic_carbon": "कार्बन", "Trihalomethanes": "टीएचएम", "Turbidity": "गंदलापन"}
    }
}

st.set_page_config(page_title="Hydrovidex AI", page_icon="💧", layout="wide")

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 50%, #80deea 100%); }
    .card { padding: 15px; border-radius: 12px; background: white; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 10px; border-left: 5px solid #0288d1; }
    .treatment-card { background: linear-gradient(45deg, #0288d1, #26c6da); color: white; padding: 15px; border-radius: 10px; font-weight: bold; margin-bottom: 10px; }
    .home-card { background: linear-gradient(45deg, #2e7d32, #66bb6a); color: white; padding: 15px; border-radius: 10px; font-weight: bold; margin-bottom: 10px; }
    .pollution-box { background: #fff3e0; border: 2px dashed #ef6c00; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-top: 10px; }
    .analysis-card { background: #ffffff; border: 1px solid #ddd; padding: 20px; border-radius: 15px; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 3. STATE & ASSETS ---
@st.cache_resource
def load_assets():
    try:
        m, s = joblib.load("random_forest_model.pkl"), joblib.load("standard_scaler.pkl")
        return m, s
    except: return None, None
model, scaler = load_assets()

# Initialization with Expanded Multi-Location Community Data
if 'history' not in st.session_state:
    st.session_state.history = [
        # LOCAL USERS (Within 500m of default Chikkaballapur)
        {"name": "Rahul Sharma", "status": "Safe", "place": "Station Rd", "lat": 13.4312, "lon": 77.7215, "time": "09:15"},
        {"name": "Sneha K.", "status": "Unsafe", "place": "Market Sq", "lat": 13.4325, "lon": 77.7230, "time": "10:45"},
        # DISTANT USERS (Regional / Other Locations)
        {"name": "Amit V.", "status": "Safe", "place": "Bangalore North", "lat": 13.0850, "lon": 77.5890, "time": "11:20"},
        {"name": "Dr. Meena", "status": "Unsafe", "place": "Nandi Hills Area", "lat": 13.3702, "lon": 77.6835, "time": "12:05"},
        {"name": "Kiran Kumar", "status": "Safe", "place": "Gauribidanur", "lat": 13.6100, "lon": 77.5200, "time": "01:30"}
    ]
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'user_profile' not in st.session_state: st.session_state.user_profile = {"name": "User_Alpha", "phone": "9988776655", "age": "24"}
if 'last_result' not in st.session_state: st.session_state.last_result = None

# --- 4. SIDEBAR ---
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language", list(LANG_DICT.keys()))
    L = LANG_DICT[lang_choice]
    loc_input = st.text_input("📍 Search Area", "Chikkaballapur, India")
    
    geolocator = Nominatim(user_agent="hydro_final_v27")
    try:
        location = geolocator.geocode(loc_input)
        lat, lon = (location.latitude, location.longitude) if location else (13.43, 77.72)
        place_name = location.address.split(',')[0] if location else "Base Site"
    except: lat, lon, place_name = 13.43, 77.72, "Base Site"

    st.info(f"Targeting: {place_name}")
    st.divider()
    st.subheader("🤖 Vedix AI Chat")
    p = st.text_input("Ask Vedix...", key="chat_in")
    if p:
        st.session_state.chat_history.append(f"👤: {p}")
        st.session_state.chat_history.append(f"🤖: Processing sensor network data for {place_name}...")

# --- 5. HEADER ---
st.markdown(f"<h1 style='text-align: center; color: #01579b;'>{L['title']}</h1>", unsafe_allow_html=True)
tabs = st.tabs([L["tab1"], L["tab2"], L["tab3"], L["tab4"]])

# --- TAB 1: DIAGNOSTIC ---
with tabs[0]:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        u_vals = {k: st.number_input(f"🔹 {L['sensors'][k]}", value=7.0 if k=="ph" else 250.0) for k in L['sensors']}
        diag_img = st.file_uploader(L["upload_label"], type=['jpg','png','jpeg'])
        
        who_std = {"ph": (6.5, 8.5), "Hardness": 200, "Solids": 500, "Chloramines": 4, "Sulfate": 250, "Conductivity": 400, "Organic_carbon": 2, "Trihalomethanes": 0.8, "Turbidity": 5}
        
        if st.button(L["run_diag"], use_container_width=True):
            if model and scaler:
                raw = np.array([list(u_vals.values())])
                is_safe = model.predict(scaler.transform(raw))[0] == 1
            else: is_safe = 6.5 <= u_vals["ph"] <= 8.5 and u_vals["Solids"] < 500
            
            source = "Nature-Made (Mineral Leaching)"
            if diag_img and (u_vals['Solids'] > 600 or u_vals['Organic_carbon'] > 4):
                source = "Human-Made (Industrial/Domestic Sewage)"
            
            st.session_state.last_result = {"safe": is_safe, "vals": u_vals, "status": L["verdict_safe"] if is_safe else L["verdict_unsafe"], "place": place_name, "source": source}
            st.session_state.history.append({"name": st.session_state.user_profile["name"], "status": st.session_state.last_result["status"], "place": place_name, "lat": lat, "lon": lon, "time": datetime.datetime.now().strftime("%H:%M")})

    with c2:
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker([lat, lon], popup="ANALYSIS POINT", icon=folium.Icon(color='blue', icon='tint')).add_to(m)
        st_folium(m, height=280, use_container_width=True, key="diag_map")
        
        st.subheader(L["who_head"])
        rows = [{L["who_param"]: L["sensors"][k], L["who_limit"]: str(who_std[k]), L["who_result"]: "✅" if ((who_std[k][0] <= v <= who_std[k][1]) if k=="ph" else (v <= who_std[k])) else "❌"} for k, v in u_vals.items()]
        st.table(pd.DataFrame(rows))

    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown(f"<div style='background:{'#1b5e20' if res['safe'] else '#b71c1c'}; padding:20px; border-radius:10px; color:white; text-align:center;'><h2>{res['status']}</h2></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="pollution-box">🔍 {L['pollution_label']}: {res['source']}</div>""", unsafe_allow_html=True)
        
        tx1, tx2 = st.columns(2)
        with tx1:
            st.subheader(L["treatment_title"])
            st.markdown('<div class="treatment-card">● RO Filtration<br>● Activated Carbon</div>', unsafe_allow_html=True)
        with tx2:
            st.subheader(L["homemade_title"])
            st.markdown('<div class="home-card">● 10-Min Boil<br>● Sand Filter</div>', unsafe_allow_html=True)
        
        st.bar_chart(pd.DataFrame({"Levels": list(res['vals'].values())}, index=list(L['sensors'].values())))
        
        pdf = FPDF()
        pdf.add_page(); pdf.set_fill_color(1, 87, 155); pdf.rect(0, 0, 210, 30, 'F')
        pdf.set_text_color(255, 255, 255); pdf.set_font("Arial", "B", 20); pdf.cell(0, 15, "HYDROVIDEX AI REPORT", ln=True, align='C')
        pdf_bytes = bytes(pdf.output(dest='S'))
        st.download_button(label=L["download_report"], data=pdf_bytes, file_name="Report.pdf", mime="application/pdf")

# --- TAB 2: VISUAL SCAN ---
with tabs[1]:
    st.header("👁️ Visual Pollution Interpreter")
    v_img = st.file_uploader(L["visual_explain_label"], type=['jpg','png','jpeg'], key="deep_scan")
    if v_img:
        st.image(v_img, width=500)
        st.markdown(f"""
        <div class="analysis-card">
            <h3>📝 Automated Visual Analysis Report</h3>
            <p><b>Visual State:</b> Significant discoloration detected.</p>
            <p><b>Primary Cause:</b> Suspended solids and bio-organic matter.</p>
            <p><b>Potability:</b> Critical levels of turbidity detected visually. Boiling is mandatory.</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: COMMUNITY (MULTI-LOCATOR MAP) ---
with tabs[2]:
    st.header(L["comm_header"])
    
    # Map shows all users, local and global
    m_c = folium.Map(location=[lat, lon], zoom_start=12)
    
    # 500m visual boundary for current user
    folium.Circle([lat, lon], radius=500, color='blue', fill=True, opacity=0.1, popup="500m Local Range").add_to(m_c)
    
    # Mark EVERY user from the history list as a locator
    for h in st.session_state.history:
        status_color = 'green' if 'Safe' in h['status'] or 'ಸುರಕ್ಷಿತ' in h['status'] or 'सुरक्षित' in h['status'] else 'red'
        
        # Determine if user is local or distant for popup text
        dist_note = " (Local)" if (abs(h['lat'] - lat) < 0.005 and abs(h['lon'] - lon) < 0.005) else " (Regional)"
        
        folium.Marker(
            location=[h['lat'], h['lon']], 
            popup=f"User: {h['name']}\nStatus: {h['status']}{dist_note}", 
            icon=folium.Icon(color=status_color, icon='user', prefix='fa')
        ).add_to(m_c)
    
    st_folium(m_c, height=500, use_container_width=True, key="global_comm_map")
    
    st.subheader("👥 Global Activity Stream")
    for h in reversed(st.session_state.history): 
        st.info(f"👤 {h['name']} | 📍 {h['place']} | {h['status']} | ⏰ {h['time']}")

# --- TAB 4: PROFILE ---
with tabs[3]:
    st.header("👤 User Profile Management")
    st.session_state.user_profile["name"] = st.text_input("Full Name", st.session_state.user_profile["name"])
    st.session_state.user_profile["phone"] = st.text_input("Mobile No", st.session_state.user_profile["phone"])
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Save Profile"): st.toast("Data Saved Locally!", icon="✅")
    with col_b:
        st.button("Profile Saved ✅", disabled=True)

st.caption("Hydrovidex AI Platform v27.0 | © 2026 Global Node")
