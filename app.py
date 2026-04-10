import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from fpdf import FPDF
import io

# --- MULTI-LANGUAGE DICTIONARY ---
LANG_DICT = {
    "English": {
        "title": "🌊 AQUAMETRIC AI : A PROFESSIONAL DIAGNOSTIC SYSTEM",
        "sidebar_title": "📍 Field Parameters",
        "locator_label": "🗺️ Live Geo-Locator (with Marker)",
        "village": "Village/City Name",
        "visual_scan": "📸 Visual Scan",
        "drop_text": "Drop sample photo here",
        "run_diag": "RUN FULL SYSTEM DIAGNOSTIC",
        "history_title": "📜 Diagnostic History",
        "chatbot_title": "🤖 AquaBot: Water Assistant",
        "verdict_safe": "BIO-SECURE (Potable)",
        "verdict_unsafe": "CONTAMINATED (Unsafe)",
        "reliability": "Reliability Index",
        "sensor_title": "🧪 Sensor Telemetry",
        "who_title": "📋 WHO Water Quality Compliance Auditor",
        "treatment_title": "🛠️ Professional Treatment Roadmap",
        "homemade_title": "🏡 Homemade Emergency Solutions",
        "ai_insight_title": "💡 AI Analytical Insight",
        "ai_insight_body": "The AI monitors **Synergistic Toxicity**. Even if individual parameters are within WHO limits, their collective interaction (the 'Cocktail Effect') may be flagged as unsafe.",
        "xai_title": "📊 XAI: Feature Influence Analysis",
        "download_report": "📥 DOWNLOAD FULL PDF REPORT",
        "bot_hi": "Hello! I am AquaBot. How can I help with your water data today?",
        "bot_high_tds": "I noticed your TDS is high. I recommend Reverse Osmosis.",
        "bot_bad_ph": "Your pH levels are outside the ideal range (6.5-8.5).",
        "bot_all_ok": "Your current parameters look mathematically stable!",
        "bot_generic": "I am monitoring your live telemetry. Click 'Run Diagnostic'.",
        "phase1": "💧 Phase 1", "phase1_desc": "Sedimentation",
        "phase2": "🧪 Phase 2", "phase2_desc": "RO Process",
        "phase3": "☀️ Phase 3", "phase3_desc": "UV Sterilization",
        "home1": "Boiling: 3 mins.", "home2": "Solar: 6 hours.", "home3": "Sand Filter.",
        "table_headers": ["Parameter", "WHO Limit", "Value", "Status"],
        "status_ok": "✅ SAFE", "status_warn": "⚠️ ALERT",
        "sensors": {"ph": "pH", "Hardness": "Hardness", "Solids": "TDS", "Chloramines": "Chloramines", "Sulfate": "Sulfate", "Conductivity": "Conductivity", "Organic_carbon": "TOC", "Trihalomethanes": "THMs", "Turbidity": "Turbidity"}
    },
    "ಕನ್ನಡ": {
        "title": "🌊 AQUAMETRIC AI : ವೃತ್ತಿಪರ ರೋಗನಿರ್ಣಯ ವ್ಯವಸ್ಥೆ",
        "sidebar_title": "📍 ಕ್ಷೇತ್ರ ನಿಯತಾಂಕಗಳು",
        "locator_label": "🗺️ ಲೈವ್ ಜಿಯೋ-ಲೋಕೇಟರ್",
        "village": "ಗ್ರಾಮ/ನಗರದ ಹೆಸರು",
        "visual_scan": "📸 ದೃಶ್ಯ ಸ್ಕ್ಯಾನ್",
        "drop_text": "ಫೋಟೋ ಹಾಕಿ",
        "run_diag": "ರೋಗನಿರ್ಣಯ ಚಲಾಯಿಸಿ",
        "history_title": "📜 ಇತಿಹಾಸ",
        "chatbot_title": "🤖 ಆಕ್ವಾಬೋಟ್",
        "verdict_safe": "ಸುರಕ್ಷಿತ",
        "verdict_unsafe": "ಅಸುರಕ್ಷಿತ",
        "reliability": "ವಿಶ್ವಾಸಾರ್ಹತೆ",
        "sensor_title": "🧪 ಟೆಲಿಮೆಟ್ರಿ",
        "who_title": "📋 WHO ಆಡಿಟರ್",
        "treatment_title": "🛠️ ಚಿಕಿತ್ಸಾ ಮಾರ್ಗಸೂಚಿ",
        "homemade_title": "🏡 ಮನೆಮದ್ದು ಪರಿಹಾರಗಳು",
        "ai_insight_title": "💡 AI ಒಳನೋಟ",
        "ai_insight_body": "ವೈಯಕ್ತಿಕ ನಿಯತಾಂಕಗಳು ಮಿತಿಯಲ್ಲಿದ್ದರೂ, ಅವುಗಳ ಒಟ್ಟು ಸಂಯೋಜನೆಯು ಅಸುರಕ್ಷಿತವಾಗಿರಬಹುದು.",
        "xai_title": "📊 XAI: ಪ್ರಭಾವದ ವಿಶ್ಲೇಷಣೆ",
        "download_report": "📥 PDF ವರದಿ",
        "bot_hi": "ನಮಸ್ಕಾರ! ನಾನು ಆಕ್ವಾಬೋಟ್.",
        "bot_high_tds": "ನಿಮ್ಮ ನೀರಿನಲ್ಲಿ TDS ಹೆಚ್ಚಿದೆ. RO ಫಿಲ್ಟರ್ ಬಳಸಿ.",
        "bot_bad_ph": "ನಿಮ್ಮ ನೀರಿನ pH ಮಟ್ಟವು ಸರಿಯಾಗಿಲ್ಲ.",
        "bot_all_ok": "ನಿಮ್ಮ ಪ್ಯಾರಾಮೀಟರ್‌ಗಳು ಸ್ಥಿರವಾಗಿವೆ!",
        "bot_generic": "ನಾನು ನಿಮ್ಮ ಡೇಟಾವನ್ನು ಗಮನಿಸುತ್ತಿದ್ದೇನೆ.",
        "phase1": "💧 ಹಂತ 1", "phase1_desc": "ಅವಕ್ಷೇಪನ",
        "phase2": "🧪 ಹಂತ 2", "phase2_desc": "RO ಶೋಧನೆ",
        "phase3": "☀️ ಹಂತ 3", "phase3_desc": "ಕ್ರಿಮಿನಾಶಕ",
        "home1": "ಕುದಿಯುವಿಕೆ: 3 ನಿಮಿಷ.", "home2": "ಸೌರ: 6 ಗಂಟೆ.", "home3": "ಮರಳು ಶೋಧಕ.",
        "table_headers": ["ನಿಯತಾಂಕ", "ಮಿತಿ", "ಮೌಲ್ಯ", "ಸ್ಥಿತಿ"],
        "status_ok": "✅ ಸುರಕ್ಷಿತ", "status_warn": "⚠️ ಎಚ್ಚರಿಕೆ",
        "sensors": {"ph": "ಪಿಹೆಚ್", "Hardness": "ಗಡಸುತನ", "Solids": "ಟಿಡಿಎಸ್", "Chloramines": "ಕ್ಲೋರಮೈನ್ಸ್", "Sulfate": "ಸಲ್ಫೇಟ್", "Conductivity": "ವಾಹಕತೆ", "Organic_carbon": "ಇಂಗಾಲ", "Trihalomethanes": "ಟಿಎಚ್‌ಎಂ", "Turbidity": "ಕಲಕುತನ"}
    },
    "हिंदी": {
        "title": "🌊 AQUAMETRIC AI : एक पेशेवर निदान प्रणाली",
        "sidebar_title": "📍 फील्ड पैरामीटर्स",
        "locator_label": "🗺️ लाइव जियो-लोकेटर",
        "village": "गांव/शहर का नाम",
        "visual_scan": "📸 विजुअल स्कैन",
        "drop_text": "फोटो यहाँ डालें",
        "run_diag": "जांच शुरू करें",
        "history_title": "📜 इतिहास",
        "chatbot_title": "🤖 एक्वाबॉट",
        "verdict_safe": "सुरक्षित",
        "verdict_unsafe": "असुरक्षित",
        "reliability": "विश्वसनीयता",
        "sensor_title": "🧪 डेटा",
        "who_title": "📋 WHO परीक्षक",
        "treatment_title": "🛠️ उपचार मार्ग",
        "homemade_title": "🏡 घरेलू समाधान",
        "ai_insight_title": "💡 AI अंतर्दृष्टि",
        "ai_insight_body": "AI सामूहिक प्रभाव की निगरानी करता है।",
        "xai_title": "📊 XAI: प्रभाव विश्लेषण",
        "download_report": "📥 PDF रिपोर्ट",
        "bot_hi": "नमस्ते! मैं एक्वाबॉट हूँ।",
        "bot_high_tds": "TDS अधिक है। RO का उपयोग करें।",
        "bot_bad_ph": "pH स्तर सही नहीं है।",
        "bot_all_ok": "पैरामीटर स्थिर दिख रहे हैं!",
        "bot_generic": "मैं आपके डेटा की निगरानी कर रहा हूँ।",
        "phase1": "💧 चरण 1", "phase1_desc": "अवसादन",
        "phase2": "🧪 चरण 2", "phase2_desc": "आरओ प्रोसेस",
        "phase3": "☀️ चरण 3", "phase3_desc": "नसबंदी",
        "home1": "उबालना: 3 मिनट.", "home2": "सोलर: 6 घंटे.", "home3": "रेत फिल्टर.",
        "table_headers": ["पैरामीटर", "सीमा", "मूल्य", "स्थिति"],
        "status_ok": "✅ सुरक्षित", "status_warn": "⚠️ चेतावनी",
        "sensors": {"ph": "पीएच", "Hardness": "कठोरता", "Solids": "टीडीएस", "Chloramines": "क्लोरामाइन", "Sulfate": "सल्फेट", "Conductivity": "चालकता", "Organic_carbon": "कार्बन", "Trihalomethanes": "टीएचएम", "Turbidity": "गंदलापन"}
    }
}

st.set_page_config(page_title="AquaMetric AI Pro", page_icon="💧", layout="wide")

# --- ASSETS LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("standard_scaler.pkl")
        return model, scaler
    except: return None, None

model, scaler = load_assets()
if 'history' not in st.session_state: st.session_state.history = []
if 'chat' not in st.session_state: st.session_state.chat = []

# --- UI STYLING ---
def inject_custom_ui():
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 50%, #80deea 100%); overflow-x: hidden; }
        [data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
        .map-container { border: 5px solid white; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .bubble { position: fixed; bottom: -100px; background: rgba(255, 255, 255, 0.4); border-radius: 50%; animation: rise 12s infinite ease-in; z-index: 0; pointer-events: none; }
        @keyframes rise { 0% { bottom: -100px; transform: translateX(0); opacity: 0.6; } 100% { bottom: 110vh; transform: translateX(100px); opacity: 0; } }
        .animated-card { padding: 20px; border-radius: 15px; margin: 10px 0; color: white; text-align: center; font-weight: bold; animation: pulse 2.5s infinite ease-in-out; }
        @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
        .phase-card { background: linear-gradient(45deg, #0288d1, #26c6da); }
        .home-card { background: linear-gradient(45deg, #43a047, #66bb6a); }
        .ai-insight-box { background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 5px; color: #0d47a1; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)
    for i in range(8):
        size, left, dur = np.random.randint(20, 80), np.random.randint(0, 95), np.random.uniform(8, 15)
        st.markdown(f'<div class="bubble" style="width:{size}px; height:{size}px; left:{left}%; animation-duration:{dur}s; animation-delay:{i*1.2}s;"></div>', unsafe_allow_html=True)

inject_custom_ui()

# --- PDF GENERATOR (FIXED) ---
def generate_pdf(user_inputs, verdict, reliability, loc_name, sensor_labels):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(33, 150, 243)
    pdf.cell(0, 15, "AQUAMETRIC AI PROFESSIONAL REPORT", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Location Analysis: {loc_name}", ln=True)
    pdf.ln(5)
    
    # Verdict Highlight
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 12, f"SYSTEM VERDICT: {verdict}", ln=True, fill=True, align='C')
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Reliability Index (AI Confidence): {reliability}", ln=True, align='C')
    pdf.ln(10)
    
    # Table Header
    pdf.set_fill_color(33, 150, 243)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(90, 10, "Parameter", 1, 0, 'C', True)
    pdf.cell(90, 10, "Recorded Value", 1, 1, 'C', True)
    
    # Table Content
    pdf.set_text_color(0, 0, 0)
    for k, v in user_inputs.items():
        pdf.cell(90, 10, str(sensor_labels[k]), 1)
        pdf.cell(90, 10, f"{v:.2f}", 1, 1)
        
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 10, "Disclaimer: This report is generated by an AI model based on provided telemetry. Please cross-verify with laboratory tests for critical infrastructure decisions.")
    
    # FIX: Byte-safe output
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    return bytes(pdf_output)

# --- SIDEBAR ---
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language / ಭಾಷೆ / भाषा", list(LANG_DICT.keys()))
    L = LANG_DICT[lang_choice]
    st.title(L["sidebar_title"])
    loc_query = st.text_input(L["village"], "Bangalore")
    uploaded_img = st.file_uploader(L["drop_text"], type=["jpg", "png", "jpeg"])
    st.divider()
    WHO_LIMITS = {"ph": (6.5, 8.5), "Hardness": (0, 200), "Solids": (0, 1000), "Chloramines": (0, 4.0), "Sulfate": (0, 250), "Conductivity": (0, 400), "Organic_carbon": (0, 2.0), "Trihalomethanes": (0, 0.08), "Turbidity": (0, 5.0)}
    user_inputs = {k: st.number_input(f"🔹 {L['sensors'][k]}", value=7.0 if k=="ph" else float(WHO_LIMITS[k][1]*0.4)) for k in WHO_LIMITS}
    
    st.subheader(L["chatbot_title"])
    for msg in st.session_state.chat[-2:]:
        st.caption(f"👤 {msg['content']}") if msg["role"]=="user" else st.info(f"🤖 {msg['content']}")
    u_input = st.text_input("Ask AquaBot...", key="bot_input")
    if u_input:
        st.session_state.chat.append({"role": "user", "content": u_input})
        res = L["bot_high_tds"] if user_inputs["Solids"] > 1000 else L["bot_generic"]
        st.session_state.chat.append({"role": "bot", "content": res}); st.rerun()

# --- MAIN DASHBOARD ---
st.title(L["title"])

# Locator
try:
    geolocator = Nominatim(user_agent="aqua_cause_final")
    location = geolocator.geocode(loc_query)
    lat, lon = (location.latitude, location.longitude) if location else (12.97, 77.59)
except: lat, lon = 12.97, 77.59

with st.container():
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    m = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker([lat, lon], popup=loc_query, icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
    st_folium(m, width="100%", height=350)
    st.markdown('</div>', unsafe_allow_html=True)

# WHO Table
st.subheader(L["who_title"])
comp_data = [[L['sensors'][k], f"{WHO_LIMITS[k][0]}-{WHO_LIMITS[k][1]}", user_inputs[k], L["status_ok"] if WHO_LIMITS[k][0] <= user_inputs[k] <= WHO_LIMITS[k][1] else L["status_warn"]] for k in WHO_LIMITS]
st.table(pd.DataFrame(comp_data, columns=L["table_headers"]))

st.markdown(f'<div class="ai-insight-box"><strong>{L["ai_insight_title"]}</strong><br>{L["ai_insight_body"]}</div>', unsafe_allow_html=True)

# RUN DIAGNOSTIC
if st.button(L["run_diag"], use_container_width=True):
    if model and scaler:
        input_data = np.array([user_inputs[k] for k in WHO_LIMITS]).reshape(1, -1)
        data_scaled = scaler.transform(input_data)
        pred = int(model.predict(data_scaled)[0])
        prob = f"{model.predict_proba(data_scaled)[0][pred]:.2%}"
        
        # XAI Chart
        st.subheader(L["xai_title"])
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = [L['sensors'][k] for k in WHO_LIMITS]
            chart_df = pd.DataFrame({"Factor": feature_names, "Influence": importances}).sort_values("Influence", ascending=True)
            st.bar_chart(chart_df, x="Factor", y="Influence", horizontal=True)
        
        # Verdict
        verdict = L["verdict_safe"] if pred == 1 else L["verdict_unsafe"]
        v_col = "#1b5e20" if pred == 1 else "#b71c1c"
        st.markdown(f"<div style='background:{v_col}; padding:25px; border-radius:15px; color:white; text-align:center;'><h1>{verdict}</h1><h3>{L['reliability']}: {prob}</h3></div>", unsafe_allow_html=True)
        
        # PDF Download
        pdf_bytes = generate_pdf(user_inputs, verdict, prob, loc_query, L['sensors'])
        st.download_button(label=L["download_report"], data=pdf_bytes, file_name=f"AquaMetric_Report_{loc_query}.pdf", mime="application/pdf")

        # Roadmap
        st.subheader(L["treatment_title"])
        cols = st.columns(3)
        for i, (p, d) in enumerate([(L["phase1"], L["phase1_desc"]), (L["phase2"], L["phase2_desc"]), (L["phase3"], L["phase3_desc"])]):
            cols[i].markdown(f'<div class="animated-card phase-card">{p}<br><small>{d}</small></div>', unsafe_allow_html=True)

        st.subheader(L["homemade_title"])
        h_cols = st.columns(3)
        for i, sol in enumerate([L["home1"], L["home2"], L["home3"]]):
            h_cols[i].markdown(f'<div class="animated-card home-card">{sol}</div>', unsafe_allow_html=True)

        st.session_state.history.insert(0, {"time": datetime.datetime.now().strftime("%H:%M"), "loc": loc_query, "ver": verdict, "img": uploaded_img, "col": v_col})

# History Display
if st.session_state.history:
    st.divider(); st.subheader(L["history_title"])
    for h in st.session_state.history[:3]:
        with st.expander(f"🕒 {h['time']} - {h['loc']}"):
            hc1, hc2 = st.columns([1, 4])
            if h["img"]: hc1.image(h["img"], use_container_width=True)
            hc2.markdown(f"**Result:** <span style='color:{h['col']}'>{h['ver']}</span>", unsafe_allow_html=True)

st.divider()
st.caption(f"AQUAMETRIC AI v7.2 | {datetime.datetime.now().year}")
