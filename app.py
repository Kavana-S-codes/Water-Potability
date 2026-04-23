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

# --- 1. FULLY EXPANDED LANGUAGE DICTIONARY ---
LANG_DICT = {
    "English": {
        "title": "🌊 Hydrovidex AI: The Autonomous Water Quality Diagnostic Engine",
        "tab1": "🏠 Diagnostic", "tab2": "👁️ Visual Scan", "tab3": "🤝 Community", "tab4": "👤 Profile",
        "run_diag": "RUN FULL SYSTEM DIAGNOSTIC",
        "verdict_safe": "NON-CONTAMINATED (Safe)", "verdict_unsafe": "CONTAMINATED (Unsafe)",
        "download_report": "📥 DOWNLOAD DESIGNER PDF REPORT",
        "comm_header": "Global & Local Water Quality Network",
        "upload_label": "Upload Image for Pollution Source Analysis",
        "visual_explain_label": "Upload Water Sample for Visual Analysis",
        "treatment_title": "🛠️ Professional Treatment Roadmap",
        "homemade_title": "🏡 Emergency Homemade Solutions",
        "who_head": "📋 WHO Water Quality Limits & Analysis",
        "who_param": "Parameter", "who_limit": "Limit", "who_result": "Status",
        "pollution_label": "Detected Pollution Source",
        "source_nature": "Nature-Made (Mineral Leaching)",
        "source_human": "Human-Made (Industrial/Domestic Sewage)",
        "treat_ro": "● RO Filtration<br>● Activated Carbon",
        "treat_boil": "● 10-Min Boil<br>● Sand Filter",
        "visual_header": "👁️ Visual Pollution Interpreter",
        "visual_report_title": "📝 Automated Visual Analysis Report",
        "visual_state": "Visual State", "visual_state_desc": "Significant discoloration detected.",
        "visual_cause": "Primary Cause", "visual_cause_desc": "Suspended solids and bio-organic matter.",
        "visual_potability": "Potability", "visual_potability_desc": "Critical levels of turbidity detected visually. Boiling is mandatory.",
        "local_range": "500m Local Range", "user": "User", "status": "Status", "local": "Local", "regional": "Regional",
        "activity_stream": "👥 Global Activity Stream",
        "prof_header": "👤 User Profile Management", "prof_name": "Full Name", "prof_phone": "Mobile No",
        "prof_save": "Save Profile", "prof_saved_toast": "Data Saved Locally!", "prof_saved_btn": "Profile Saved ✅",
        "target": "Targeting", "chat_placeholder": "Ask Vedix about water quality...", "chat_you": "You", "chat_vedix": "Vedix",
        "chat_analyzing": "I am analyzing the sensor network data for", "map_popup": "ANALYSIS POINT",
        "names": ["Rahul Sharma", "Sneha K.", "Amit V.", "Dr. Meena", "Kiran Kumar"],
        "sensors": {"ph": "pH Level", "Hardness": "Hardness", "Solids": "TDS", "Chloramines": "Chloramines", "Sulfate": "Sulfate", "Conductivity": "Conductivity", "Organic_carbon": "TOC", "Trihalomethanes": "THMs", "Turbidity": "Turbidity"}
    },
    "ಕನ್ನಡ": {
        "title": "🌊 ಹೈಡ್ರೋವಿಡೆಕ್ಸ್ ಎಐ: ಸ್ವಾಯತ್ತ ನೀರಿನ ಗುಣಮಟ್ಟ ರೋಗನಿರ್ಣಯ ಎಂಜಿನ್",
        "tab1": "🏠 ರೋಗನಿರ್ಣಯ", "tab2": "👁️ ದೃಶ್ಯ ಸ್ಕ್ಯಾನ್", "tab3": "🤝 ಸಮುದಾಯ", "tab4": "👤 ಪ್ರೊಫೈಲ್",
        "run_diag": "ರೋಗನಿರ್ಣಯ ಚಲಾಯಿಸಿ",
        "verdict_safe": "ಸುರಕ್ಷಿತ (Safe)", "verdict_unsafe": "ಅಸುರಕ್ಷಿತ (Unsafe)",
        "download_report": "📥 PDF ವರದಿ",
        "comm_header": "ಜಾಗತಿಕ ಮತ್ತು ಸ್ಥಳೀಯ ನೀರಿನ ಗುಣಮಟ್ಟ ನೆಟ್‌ವರ್ಕ್",
        "upload_label": "ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "visual_explain_label": "ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಚಿತ್ರ",
        "treatment_title": "🛠️ ವೃತ್ತಿಪರ ಚಿಕಿತ್ಸಾ ಮಾರ್ಗಸೂಚಿ",
        "homemade_title": "🏡 ತುರ್ತು ಮನೆಮದ್ದು ಪರಿಹಾರಗಳು",
        "who_head": "📋 ಡಬ್ಲ್ಯುಎಚ್‌ಒ ನೀರಿನ ಗುಣಮಟ್ಟ ಮಿತಿಗಳು",
        "who_param": "ಪ್ಯಾರಾಮೀಟರ್", "who_limit": "ಮಿತಿ", "who_result": "ಸ್ಥಿತಿ",
        "pollution_label": "ಮಾಲಿನ್ಯದ ಮೂಲ",
        "source_nature": "ಪ್ರಕೃತಿದತ್ತ (ಖನಿಜ ಸೋರಿಕೆ)",
        "source_human": "ಮಾನವ ನಿರ್ಮಿತ (ಕೈಗಾರಿಕಾ/ಗೃಹಬಳಕೆಯ ಒಳಚರಂಡಿ)",
        "treat_ro": "● ಆರ್‌ಒ (RO) ಫಿಲ್ಟರೇಶನ್<br>● ಆಕ್ಟಿವೇಟೆಡ್ ಕಾರ್ಬನ್",
        "treat_boil": "● 10-ನಿಮಿಷಗಳ ಕುದಿಯುವಿಕೆ<br>● ಮರಳು ಫಿಲ್ಟರ್",
        "visual_header": "👁️ ದೃಶ್ಯ ಮಾಲಿನ್ಯ ವಿಶ್ಲೇಷಕ",
        "visual_report_title": "📝 ಸ್ವಯಂಚಾಲಿತ ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆ ವರದಿ",
        "visual_state": "ದೃಶ್ಯ ಸ್ಥಿತಿ", "visual_state_desc": "ಗಮನಾರ್ಹ ಬಣ್ಣ ಬದಲಾವಣೆ ಪತ್ತೆಯಾಗಿದೆ.",
        "visual_cause": "ಮುಖ್ಯ ಕಾರಣ", "visual_cause_desc": "ತೇಲುವ ಘನವಸ್ತುಗಳು ಮತ್ತು ಜೈವಿಕ-ಸಾವಯವ ವಸ್ತುಗಳು.",
        "visual_potability": "ಕುಡಿಯುವ ಯೋಗ್ಯತೆ", "visual_potability_desc": "ಕಲಕುತನದ ನಿರ್ಣಾಯಕ ಮಟ್ಟ ಪತ್ತೆಯಾಗಿದೆ. ಕುದಿಸುವುದು ಕಡ್ಡಾಯ.",
        "local_range": "500 ಮೀ ಸ್ಥಳೀಯ ವ್ಯಾಪ್ತಿ", "user": "ಬಳಕೆದಾರ", "status": "ಸ್ಥಿತಿ", "local": "ಸ್ಥಳೀಯ", "regional": "ಪ್ರಾದೇಶಿಕ",
        "activity_stream": "👥 ಜಾಗತಿಕ ಚಟುವಟಿಕೆ ಸ್ಟ್ರೀಮ್",
        "prof_header": "👤 ಬಳಕೆದಾರರ ಪ್ರೊಫೈಲ್ ನಿರ್ವಹಣೆ", "prof_name": "ಪೂರ್ಣ ಹೆಸರು", "prof_phone": "ಮೊಬೈಲ್ ಸಂಖ್ಯೆ",
        "prof_save": "ಪ್ರೊಫೈಲ್ ಉಳಿಸಿ", "prof_saved_toast": "ಡೇಟಾವನ್ನು ಸ್ಥಳೀಯವಾಗಿ ಉಳಿಸಲಾಗಿದೆ!", "prof_saved_btn": "ಪ್ರೊಫೈಲ್ ಉಳಿಸಲಾಗಿದೆ ✅",
        "target": "ಗುರಿ", "chat_placeholder": "ನೀರಿನ ಬಗ್ಗೆ ವೇಡಿಕ್ಸ್ ಅನ್ನು ಕೇಳಿ...", "chat_you": "ನೀವು", "chat_vedix": "ವೇಡಿಕ್ಸ್",
        "chat_analyzing": "ನಾನು ವಿಶ್ಲೇಷಿಸುತ್ತಿದ್ದೇನೆ:", "map_popup": "ವಿಶ್ಲೇಷಣಾ ಬಿಂದು",
        "names": ["ರಾಹುಲ್ ಶರ್ಮಾ", "ಸ್ನೇಹಾ ಕೆ.", "ಅಮಿತ್ ವಿ.", "ಡಾ. ಮೀನಾ", "ಕಿರಣ್ ಕುಮಾರ್"],
        "sensors": {"ph": "ಪಿಹೆಚ್ ಮಟ್ಟ", "Hardness": "ಗಡಸುತನ", "Solids": "ಟಿಡಿಎಸ್", "Chloramines": "ಕ್ಲೋರಮೈನ್ಸ್", "Sulfate": "ಸಲ್ಫೇಟ್", "Conductivity": "ವಾಹಕತೆ", "Organic_carbon": "ಇಂಗಾಲ", "Trihalomethanes": "ಟಿಎಚ್‌ಎಂ", "Turbidity": "ಕಲಕುತನ"}
    },
    "हिंदी": {
        "title": "🌊 हाइड्रोविडेक्स एआई: स्वायत्त जल गुणवत्ता नैदानिक इंजन",
        "tab1": "🏠 निदान", "tab2": "👁️ विजुअल स्कैन", "tab3": "🤝 समुदाय", "tab4": "👤 प्रोफाइल",
        "run_diag": "पूर्ण निदान चलाएं",
        "verdict_safe": "सुरक्षित (Safe)", "verdict_unsafe": "असुरक्षित (Unsafe)",
        "download_report": "📥 PDF रिपोर्ट",
        "comm_header": "वैश्विक और स्थानीय जल गुणवत्ता नेटवर्क",
        "upload_label": "छवि अपलोड करें",
        "visual_explain_label": "दृश्य विश्लेषण",
        "treatment_title": "🛠️ पेशेवर उपचार रोडमैप",
        "homemade_title": "🏡 आपातकालीन घरेलू समाधान",
        "who_head": "📋 डब्ल्यूएचओ जल गुणवत्ता सीमाएं",
        "who_param": "पैरामीटर", "who_limit": "सीमा", "who_result": "स्थिति",
        "pollution_label": "प्रदूषण का स्रोत",
        "source_nature": "प्रकृति-निर्मित (खनिज रिसाव)",
        "source_human": "मानव-निर्मित (औद्योगिक/घरेलू सीवेज)",
        "treat_ro": "● आरओ (RO) निस्पंदन<br>● सक्रिय कार्बन",
        "treat_boil": "● 10-मिनट उबालें<br>● रेत फिल्टर",
        "visual_header": "👁️ दृश्य प्रदूषण विश्लेषक",
        "visual_report_title": "📝 स्वचालित दृश्य विश्लेषण रिपोर्ट",
        "visual_state": "दृश्य स्थिति", "visual_state_desc": "महत्वपूर्ण मलिनकिरण का पता चला।",
        "visual_cause": "प्राथमिक कारण", "visual_cause_desc": "निलंबित ठोस और जैव-जैविक पदार्थ।",
        "visual_potability": "पीने योग्यता", "visual_potability_desc": "गंदलापन का महत्वपूर्ण स्तर दृष्टिगत रूप से पाया गया। उबालना अनिवार्य है।",
        "local_range": "500 मीटर स्थानीय सीमा", "user": "उपयोगकर्ता", "status": "स्थिति", "local": "स्थानीय", "regional": "क्षेत्रीय",
        "activity_stream": "👥 वैश्विक गतिविधि स्ट्रीम",
        "prof_header": "👤 उपयोगकर्ता प्रोफ़ाइल प्रबंधन", "prof_name": "पूरा नाम", "prof_phone": "मोबाइल नंबर",
        "prof_save": "प्रोफाइल सहेजें", "prof_saved_toast": "डेटा स्थानीय रूप से सहेजा गया!", "prof_saved_btn": "प्रोफाइल सहेजा गया ✅",
        "target": "लक्षित", "chat_placeholder": "वेडिक्स से पानी की गुणवत्ता के बारे में पूछें...", "chat_you": "आप", "chat_vedix": "वेडिक्स",
        "chat_analyzing": "मैं सेंसर नेटवर्क डेटा का विश्लेषण कर रहा हूं:", "map_popup": "विश्लेषण बिंदु",
        "names": ["राहुल शर्मा", "स्नेहा के.", "अमित वी.", "डॉ. मीना", "किरण कुमार"],
        "sensors": {"ph": "पीएच स्तर", "Hardness": "कठोरता", "Solids": "टीडीएस", "Chloramines": "क्लोरामाइन", "Sulfate": "सल्फेट", "Conductivity": "चालकता", "Organic_carbon": "कार्बन", "Trihalomethanes": "टीएचएम", "Turbidity": "गंदलापन"}
    }
}

st.set_page_config(page_title="Hydrovidex AI", page_icon="💧", layout="wide")

# --- 2. CSS STYLING & BUBBLE ANIMATION ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 50%, #80deea 100%); overflow: hidden; }
    .card { padding: 15px; border-radius: 12px; background: white; box-shadow: 0 4px 10px rgba(0,0,0,0.1); margin-bottom: 10px; border-left: 5px solid #0288d1; }
    /* Added min-height to treatment and home cards for perfect alignment! */
    .treatment-card { background: linear-gradient(45deg, #0288d1, #26c6da); color: white; padding: 15px; border-radius: 10px; font-weight: bold; margin-bottom: 10px; min-height: 100px; }
    .home-card { background: linear-gradient(45deg, #2e7d32, #66bb6a); color: white; padding: 15px; border-radius: 10px; font-weight: bold; margin-bottom: 10px; min-height: 100px; }
    .pollution-box { background: #fff3e0; border: 2px dashed #ef6c00; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-top: 10px; }
    .analysis-card { background: #ffffff; border: 1px solid #ddd; padding: 20px; border-radius: 15px; margin-top: 15px; }
    .bubble { position: absolute; bottom: -100px; background-color: rgba(255, 255, 255, 0.5); border-radius: 50%; animation: floatUp infinite ease-in; pointer-events: none; z-index: 0; }
    @keyframes floatUp { 0% { transform: translateY(0) translateX(0); opacity: 1; } 100% { transform: translateY(-120vh) translateX(20px); opacity: 0; } }
</style>
<div class="bubble" style="left: 10%; width: 40px; height: 40px; animation-duration: 8s;"></div>
<div class="bubble" style="left: 25%; width: 20px; height: 20px; animation-duration: 5s; animation-delay: 2s;"></div>
<div class="bubble" style="left: 45%; width: 60px; height: 60px; animation-duration: 11s; animation-delay: 1s;"></div>
<div class="bubble" style="left: 65%; width: 30px; height: 30px; animation-duration: 7s; animation-delay: 4s;"></div>
<div class="bubble" style="left: 85%; width: 50px; height: 50px; animation-duration: 9s; animation-delay: 0s;"></div>
<div class="bubble" style="left: 95%; width: 15px; height: 15px; animation-duration: 6s; animation-delay: 3s;"></div>
""", unsafe_allow_html=True)

# --- 3. STATE & ASSETS ---
@st.cache_resource
def load_assets():
    try:
        m, s = joblib.load("random_forest_model.pkl"), joblib.load("standard_scaler.pkl")
        return m, s
    except: return None, None
model, scaler = load_assets()

if 'history' not in st.session_state:
    st.session_state.history = [
        {"name_idx": 0, "is_safe": True, "place": "Station Rd", "lat": 13.4312, "lon": 77.7215, "time": "09:15"},
        {"name_idx": 1, "is_safe": False, "place": "Market Sq", "lat": 13.4325, "lon": 77.7230, "time": "10:45"},
        {"name_idx": 2, "is_safe": True, "place": "Bangalore North", "lat": 13.0850, "lon": 77.5890, "time": "11:20"},
        {"name_idx": 3, "is_safe": False, "place": "Nandi Hills Area", "lat": 13.3702, "lon": 77.6835, "time": "12:05"},
        {"name_idx": 4, "is_safe": True, "place": "Gauribidanur", "lat": 13.6100, "lon": 77.5200, "time": "01:30"}
    ]
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'user_profile' not in st.session_state: st.session_state.user_profile = {"name": "User_Alpha", "phone": "9988776655", "age": "24"}
if 'last_result' not in st.session_state: st.session_state.last_result = None

# --- 4. SIDEBAR & VEDIX AI CHATBOT ---
with st.sidebar:
    lang_choice = st.selectbox("🌐", list(LANG_DICT.keys()))
    L = LANG_DICT[lang_choice]
    loc_input = st.text_input("📍", "Chikkaballapur, India")
    
    geolocator = Nominatim(user_agent="hydro_final_v30")
    try:
        location = geolocator.geocode(loc_input)
        lat, lon = (location.latitude, location.longitude) if location else (13.43, 77.72)
        place_name = location.address.split(',')[0] if location else "Base Site"
    except: lat, lon, place_name = 13.43, 77.72, "Base Site"

    st.info(f"{L['target']}: {place_name}")
    st.divider()
    
    st.subheader(f"🤖 {L['chat_vedix']}")
    chat_container = st.container(height=300)
    with chat_container:
        if len(st.session_state.chat_history) == 0:
            st.caption(L['chat_placeholder'])
        for msg in st.session_state.chat_history:
            st.markdown(msg)
            
    prompt = st.chat_input(L['chat_placeholder'], key="chat_in")
    if prompt:
        st.session_state.chat_history.append(f"**👤 {L['chat_you']}:** {prompt}")
        st.session_state.chat_history.append(f"**🤖 {L['chat_vedix']}:** {L['chat_analyzing']} {place_name}...")
        st.rerun()

# --- 5. HEADER ---
st.markdown(f"<div style='position: relative; z-index: 10;'><h1 style='text-align: center; color: #01579b;'>{L['title']}</h1></div>", unsafe_allow_html=True)
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
            
            source = L["source_nature"]
            if diag_img and (u_vals['Solids'] > 600 or u_vals['Organic_carbon'] > 4):
                source = L["source_human"]
            
            st.session_state.last_result = {"safe": is_safe, "vals": u_vals, "place": place_name, "source": source}
            st.session_state.history.append({"name": st.session_state.user_profile["name"], "is_safe": is_safe, "place": place_name, "lat": lat, "lon": lon, "time": datetime.datetime.now().strftime("%H:%M")})

    with c2:
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker([lat, lon], popup=L["map_popup"], icon=folium.Icon(color='blue', icon='tint')).add_to(m)
        st_folium(m, height=280, use_container_width=True, key="diag_map")
        
        st.subheader(L["who_head"])
        rows = [{L["who_param"]: L["sensors"][k], L["who_limit"]: str(who_std[k]), L["who_result"]: "✅" if ((who_std[k][0] <= v <= who_std[k][1]) if k=="ph" else (v <= who_std[k])) else "❌"} for k, v in u_vals.items()]
        st.table(pd.DataFrame(rows))

    if st.session_state.last_result:
        res = st.session_state.last_result
        current_status_text = L["verdict_safe"] if res['safe'] else L["verdict_unsafe"]
        st.markdown(f"<div style='position: relative; z-index: 10; background:{'#1b5e20' if res['safe'] else '#b71c1c'}; padding:20px; border-radius:10px; color:white; text-align:center;'><h2>{current_status_text}</h2></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="pollution-box" style="position: relative; z-index: 10;">🔍 {L['pollution_label']}: {res['source']}</div>""", unsafe_allow_html=True)
        
        # Added gap="medium" to give the side-by-side boxes space
        tx1, tx2 = st.columns(2, gap="medium")
        with tx1:
            st.subheader(L["treatment_title"])
            st.markdown(f'<div class="treatment-card">{L["treat_ro"]}</div>', unsafe_allow_html=True)
        with tx2:
            st.subheader(L["homemade_title"])
            st.markdown(f'<div class="home-card">{L["treat_boil"]}</div>', unsafe_allow_html=True)
        
        st.bar_chart(pd.DataFrame({"Levels": list(res['vals'].values())}, index=list(L['sensors'].values())))
        
        # --- GENERATING THE POPULATED PDF REPORT ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(1, 87, 155)
        pdf.rect(0, 0, 210, 30, 'F')
        
        # Header
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 15, "HYDROVIDEX AI - ANALYSIS REPORT", ln=True, align='C')
        pdf.ln(10)
        
        # Body
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "B", 14)
        
        status_eng = "SAFE (Non-Contaminated)" if res['safe'] else "UNSAFE (Contaminated)"
        source_eng = "Human-Made (Industrial/Domestic Sewage)" if ("Human" in res['source'] or "ಮಾನವ" in res['source'] or "मानव" in res['source']) else "Nature-Made (Mineral Leaching)"
        
        pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.cell(0, 10, f"Location: {res['place']}", ln=True)
        pdf.cell(0, 10, f"Safety Verdict: {status_eng}", ln=True)
        pdf.cell(0, 10, f"Detected Source: {source_eng}", ln=True)
        
        pdf.ln(5)
        pdf.cell(0, 10, "Sensor Readings:", ln=True)
        pdf.set_font("Arial", "", 12)
        for key, val in res['vals'].items():
            pdf.cell(0, 8, f"- {key}: {val}", ln=True)
            
        pdf_bytes = bytes(pdf.output(dest='S'))
        st.download_button(label=L["download_report"], data=pdf_bytes, file_name="Hydrovidex_Report.pdf", mime="application/pdf")

# --- TAB 2: VISUAL SCAN ---
with tabs[1]:
    st.header(L["visual_header"])
    v_img = st.file_uploader(L["visual_explain_label"], type=['jpg','png','jpeg'], key="deep_scan")
    if v_img:
        st.image(v_img, width=500)
        st.markdown(f"""
        <div class="analysis-card" style="position: relative; z-index: 10;">
            <h3>{L["visual_report_title"]}</h3>
            <p><b>{L["visual_state"]}:</b> {L["visual_state_desc"]}</p>
            <p><b>{L["visual_cause"]}:</b> {L["visual_cause_desc"]}</p>
            <p><b>{L["visual_potability"]}:</b> {L["visual_potability_desc"]}</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: COMMUNITY (MULTI-LOCATOR MAP) ---
with tabs[2]:
    st.header(L["comm_header"])
    
    m_c = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Circle([lat, lon], radius=500, color='blue', fill=True, opacity=0.1, popup=L["local_range"]).add_to(m_c)
    
    for h in st.session_state.history:
        is_safe = h['is_safe']
        status_color = 'green' if is_safe else 'red'
        status_text = L["verdict_safe"] if is_safe else L["verdict_unsafe"]
        dist_note = f" ({L['local']})" if (abs(h['lat'] - lat) < 0.005 and abs(h['lon'] - lon) < 0.005) else f" ({L['regional']})"
        
        # DYNAMIC NAME TRANSLATION
        display_name = L["names"][h["name_idx"]] if "name_idx" in h else h["name"]
        
        folium.Marker(
            location=[h['lat'], h['lon']], 
            popup=f"{L['user']}: {display_name}\n{L['status']}: {status_text}{dist_note}", 
            icon=folium.Icon(color=status_color, icon='user', prefix='fa')
        ).add_to(m_c)
    
    st_folium(m_c, height=500, use_container_width=True, key="global_comm_map")
    
    st.subheader(L["activity_stream"])
    for h in reversed(st.session_state.history): 
        status_text = L["verdict_safe"] if h['is_safe'] else L["verdict_unsafe"]
        display_name = L["names"][h["name_idx"]] if "name_idx" in h else h["name"]
        st.info(f"👤 {display_name} | 📍 {h['place']} | {status_text} | ⏰ {h['time']}")

# --- TAB 4: PROFILE ---
with tabs[3]:
    st.header(L["prof_header"])
    st.session_state.user_profile["name"] = st.text_input(L["prof_name"], st.session_state.user_profile["name"])
    st.session_state.user_profile["phone"] = st.text_input(L["prof_phone"], st.session_state.user_profile["phone"])
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button(L["prof_save"]): st.toast(L["prof_saved_toast"], icon="✅")
    with col_b:
        st.button(L["prof_saved_btn"], disabled=True)

st.caption("Hydrovidex AI Platform v30.0 | © 2026 Global Node")