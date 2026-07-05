import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from fpdf import FPDF
import cv2
import os
import random 

# --- 1. COMPREHENSIVE LANGUAGE DICTIONARY ---
LANG_DICT = {
    "English": {
        "title": "🌊 Hydrovidex AI: The Autonomous Water Quality Diagnostic Engine",
        "tab1": "🏠 Diagnostic", "tab2": "👁️ Visual Scan", "tab3": "🤝 Community", "tab4": "👤 Profile",
        "run_diag": "RUN FULL SYSTEM DIAGNOSTIC",
        "verdict_safe": "NON-CONTAMINATED (Safe)", "verdict_unsafe": "CONTAMINATED (Unsafe)",
        "download_report": "📥 DOWNLOAD DESIGNER PDF REPORT",
        "comm_header": "Global & Local Water Quality Network",
        "upload_label": "Upload Image for Origin Analysis (Human vs Nature)",
        "visual_explain_label": "Upload Sample for Microplastic & Origin Scan",
        "treatment_title": "🛠️ Professional Treatment Roadmap",
        "homemade_title": "🏡 Emergency Homemade Solutions",
        "treatment_list": "● RO Filtration<br>● UV Sterilization<br>● Activated Carbon",
        "homemade_list": "● 10-Min Boil<br>● Sand Filter<br>● Solar Disinfection",
        "graph_title": "📊 Parameter Concentration Analysis",
        "pollution_label": "Detected Pollution Source",
        "stream_head": "👥 Community Activity & Feedback",
        "prof_head": "👤 User Profile Management",
        "save_prof": "Save Profile Settings",
        "age_label": "Age", "addr_label": "Address", "name_label": "Full Name", "phone_label": "Mobile",
        "who_head": "📋 WHO Water Quality Limits & Analysis",
        "who_param": "Parameter", "who_limit": "Limit", "who_result": "Status",
        "source_industrial": "Human-Made (Industrial/Domestic Sewage)",
        "source_natural": "Nature-Made (Mineral Leaching)",
        "sensors": {"ph": "pH Level", "Hardness": "Hardness", "Solids": "TDS", "Chloramines": "Chloramines", "Sulfate": "Sulfate", "Conductivity": "Conductivity", "Organic_carbon": "TOC", "Trihalomethanes": "THMs", "Turbidity": "Turbidity"}
    },
    "ಕನ್ನಡ": {
        "title": "🌊 ಹೈಡ್ರೋವಿಡೆಕ್ಸ್ ಎಐ: ಸ್ವಾಯತ್ತ ನೀರಿನ ಗುಣಮಟ್ಟದ ಎಂಜಿನ್",
        "tab1": "🏠 ರೋಗನಿರ್ಣಯ", "tab2": "👁️ ದೃಶ್ಯ ಸ್ಕ್ಯಾನ್", "tab3": "🤝 ಸಮುದಾಯ", "tab4": "👤 ಪ್ರೊಫೈಲ್",
        "run_diag": "ಪೂರ್ಣ ರೋಗನಿರ್ಣಯ ಚಲಾಯಿಸಿ",
        "verdict_safe": "ಸುರಕ್ಷಿತ (ಕಲುಷಿತಗೊಂಡಿಲ್ಲ)", "verdict_unsafe": "ಅಸುರಕ್ಷಿತ (ಕಲುಷಿತಗೊಂಡಿದೆ)",
        "download_report": "📥 PDF ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "comm_header": "ಜಾಗತಿಕ ಮತ್ತು ಸ್ಥಳೀಯ ಜಾಲ",
        "upload_label": "ಮೂಲ ವಿಶ್ಲೇಷಣೆಗಾಗಿ ಚಿತ್ರ (ಮಾನವ ಅಥವಾ ನಿಸರ್ಗ)",
        "visual_explain_label": "ಸೂಕ್ಷ್ಮ ಪ್ಲಾಸ್ಟಿಕ್ ಮತ್ತು ಮೂಲ ಸ್ಕ್ಯಾನ್‌ಗಾಗಿ ಚಿತ್ರ",
        "treatment_title": "🛠️ ವೃತ್ತಿಪರ ಸಂಸ್ಕರಣಾ ಮಾರ್ಗಸೂಚಿ",
        "homemade_title": "🏡 ತುರ್ತು ಮನೆಮದ್ದು ಪರಿಹಾರಗಳು",
        "treatment_list": "● ಆರ್‌ಒ ಫಿಲ್ಟರೇಶನ್<br>● ಯುವಿ ಕ್ರಿಮಿನಾಶಕ<br>● ಸಕ್ರಿಯ ಇಂಗಾಲ",
        "homemade_list": "● 10 ನಿಮಿಷ ಕುದಿಸಿ<br>● ಮರಳು ಫಿಲ್ಟರ್<br>● ಸೌರ ಸೋಂಕುಗಳೆತ",
        "graph_title": "📊 ನಿಯತಾಂಕ ಸಾಂದ್ರತೆ",
        "who_head": "📋 WHO ನೀರಿನ ಗುಣಮಟ್ಟದ ವಿಶ್ಲೇಷಣೆ",
        "who_param": "ನಿಯತಾಂಕ", "who_limit": "ಮಿತಿ", "who_result": "ಸ್ಥಿತಿ",
        "pollution_label": "ಪತ್ತೆಯಾದ ಮಾಲಿನ್ಯದ ಮೂಲ",
        "stream_head": "👥 ಸಮುದಾಯ ಚಟುವಟಿಕೆ ಮತ್ತು ಪ್ರತಿಕ್ರಿಯೆ",
        "prof_head": "👤 ಬಳಕೆದಾರರ ಪ್ರೊಫೈಲ್",
        "save_prof": "ಪ್ರೊಫೈಲ್ ಉಳಿಸಿ",
        "age_label": "ವಯಸ್ಸು", "addr_label": "ವಿಳಾಸ", "name_label": "ಪೂರ್ಣ ಹೆಸರು", "phone_label": "ಮೊಬೈಲ್ ಸಂಖ್ಯೆ",
        "source_industrial": "ಮಾನವ ನಿರ್ಮಿತ (ಕೈಗಾರಿಕಾ/ಮನೆಯ ಒಳಚರಂಡಿ)",
        "source_natural": "ನಿಸರ್ಗ ನಿರ್ಮಿತ (ಖನಿಜ ಸೋರುವಿಕೆ)",
        "sensors": {"ph": "ಪಿಹೆಚ್ ಮಟ್ಟ", "Hardness": "ಗಡಸುತನ", "Solids": "ಟಿಡಿಎಸ್", "Chloramines": "ಕ್ಲೋರಮೈನ್ಸ್", "Sulfate": "ಸಲ್ಫೇಟ್", "Conductivity": "ವಾಹಕತೆ", "Organic_carbon": "ಇಂಗಾಲ", "Trihalomethanes": "ಟಿಎಚ್‌ಎಂ", "Turbidity": "ಕಲಕುತನ"}
    },
    "हिंदी": {
        "title": "🌊 हाइड्रोविडेक्स एआई: स्वायत्त जल गुणवत्ता इंजन",
        "tab1": "🏠 निदान", "tab2": "👁️ विजुअल स्कैन", "tab3": "🤝 समुदाय", "tab4": "👤 प्रोफाइल",
        "run_diag": "पूर्ण निदान चलाएं",
        "verdict_safe": "सुरक्षित (गैर-प्रदूषित)", "verdict_unsafe": "असुरक्षित (प्रदूषित)",
        "download_report": "📥 पीडीएफ रिपोर्ट डाउनलोड करें",
        "comm_header": "ग्लोबल और लोकल नेटवर्क",
        "upload_label": "स्रोत विश्लेषण के लिए छवि (मानव बनाम प्रकृति)",
        "visual_explain_label": "माइक्रोप्लास्टिक और स्रोत स्कैन के लिए छवि",
        "treatment_title": "🛠️ प्रोफेशनल उपचार रोडमैप",
        "homemade_title": "🏡 घरेलू समाधान",
        "treatment_list": "● आरओ फिल्ट्रेशन<br>● यूवी नसबंदी<br>● सक्रिय कार्बन",
        "homemade_list": "● 10-मिनट उबालें<br>● रेत फिल्टर<br>● सौर कीटाणुशोधन",
        "pollution_label": "प्रदूषण का स्रोत",
        "stream_head": "👥 समुदाय गतिविधि और प्रतिक्रिया",
        "prof_head": "👤 उपयोगकर्ता प्रोफ़ाइल",
        "save_prof": "प्रोफ़ाइल सहेजें",
        "age_label": "आयु", "addr_label": "पता", "name_label": "पूरा नाम", "phone_label": "मोबाइल नंबर",
        "who_head": "📋 WHO जल गुणवत्ता विश्लेषण",
        "who_param": "पैरामीटर", "who_limit": "सीमा", "who_result": "स्थिति",
        "source_industrial": "मानव निर्मित (औद्योगिक/घरेलू सीवेज)",
        "source_natural": "प्रकृति निर्मित (खनिज निक्षालन)",
        "sensors": {"ph": "पीएच स्तर", "Hardness": "कठोरता", "Solids": "टीडीएस", "Chloramines": "क्लोरामाइन", "Sulfate": "सल्फे़ट", "Conductivity": "चालकता", "Organic_carbon": "कार्बन", "Trihalomethanes": "टीएचएम", "Turbidity": "गंदलापन"}
    }
}

st.set_page_config(page_title="Hydrovidex AI v27", page_icon="💧", layout="wide")

# --- 2. CSS & BUBBLE ANIMATION ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 50%, #80deea 100%); }
    .treatment-card { background: linear-gradient(45deg, #0288d1, #26c6da); color: white; padding: 15px; border-radius: 10px; font-weight: bold; margin-bottom: 10px; }
    .home-card { background: linear-gradient(45deg, #2e7d32, #66bb6a); color: white; padding: 15px; border-radius: 10px; font-weight: bold; margin-bottom: 10px; }
    .pollution-box { background: #fff3e0; border: 2px dashed #ef6c00; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-top: 10px; }
    .analysis-card { background: white; padding: 20px; border-radius: 15px; border: 1px solid #0288d1; margin-top: 15px; }
    .comm-card { background: rgba(255,255,255,0.7); border-radius: 12px; padding: 15px; margin-bottom: 10px; border-left: 5px solid #0288d1; }
    #bubbles { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; }
</style>
<canvas id="bubbles"></canvas>
<script>
    const canvas = document.getElementById('bubbles');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth; canvas.height = window.innerHeight;
    let particles = [];
    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = canvas.height + Math.random() * 100;
            this.size = Math.random() * 6 + 2;
            this.speed = Math.random() * 0.5 + 0.2;
            this.opacity = Math.random() * 0.4;
        }
        update() { this.y -= this.speed; if (this.y < -20) { this.y = canvas.height + 20; this.x = Math.random() * canvas.width; } }
        draw() { ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`; ctx.beginPath(); ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2); ctx.fill(); }
    }
    function init() { for (let i = 0; i < 60; i++) { particles.push(new Particle()); } }
    function animate() { ctx.clearRect(0, 0, canvas.width, canvas.height); particles.forEach(p => { p.update(); p.draw(); }); requestAnimationFrame(animate); }
    init(); animate();
</script>
""", unsafe_allow_html=True)

# --- 3. STATE & PKL LOADING ---
@st.cache_resource
def load_assets():
    try:
        m, s = joblib.load("random_forest_model.pkl"), joblib.load("standard_scaler.pkl")
        return m, s
    except: return None, None
model, scaler = load_assets()

if 'history' not in st.session_state:
    st.session_state.history = [
        {"name": "Rahul Sharma", "age": "29", "phone": "9845012345", "address": "MG Road, Blr", "status": "Safe", "lat": 13.431, "lon": 77.721, "place": "Station Rd", "time": "09:15", "feedback": "System is very accurate!"}
    ]
if 'messages' not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "I am Vedix AI. How can I assist you?"}]
if 'user_profile' not in st.session_state: st.session_state.user_profile = {"name": "User_Alpha", "phone": "9988776655", "age": "24", "address": "Node 1"}
if 'last_result' not in st.session_state: st.session_state.last_result = None

# --- 4. SIDEBAR CHATBOT ---
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language", list(LANG_DICT.keys()))
    L = LANG_DICT[lang_choice]
    loc_input = st.text_input("📍 Search Area", "Chikkaballapur, India")
    geolocator = Nominatim(user_agent="hydro_v27_final")
    try:
        location = geolocator.geocode(loc_input)
        lat, lon = (location.latitude, location.longitude) if location else (13.43, 77.72)
        place_name = location.address.split(',')[0] if location else "Base Site"
    except: lat, lon, place_name = 13.43, 77.72, "Base Site"
    
    st.divider()
    st.subheader("🤖 Vedix AI Chatbot")
    q_col1, q_col2 = st.columns(2)
    if q_col1.button("What is TDS?"): st.session_state.messages.append({"role": "user", "content": "What is TDS?"})
    if q_col2.button("Safe pH?"): st.session_state.messages.append({"role": "user", "content": "What is a safe pH?"})
    
    chat_box = st.container(height=200)
    for m in st.session_state.messages: chat_box.chat_message(m["role"]).write(m["content"])
    if p := st.chat_input("Ask Vedix..."):
        st.session_state.messages.append({"role": "user", "content": p})
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.markdown(f"<h1 style='text-align: center; color: #01579b;'>{L['title']}</h1>", unsafe_allow_html=True)
tabs = st.tabs([L["tab1"], L["tab2"], L["tab3"], L["tab4"]])

# --- TAB 1: DIAGNOSTIC ---
with tabs[0]:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        u_vals = {k: st.number_input(f"🔹 {L['sensors'][k]}", value=7.0 if k=="ph" else 250.0, key=f"d_{k}") for k in L['sensors']}
        diag_source_img = st.file_uploader(L["upload_label"], type=['jpg','png','jpeg'], key="diag_source")
        
        if st.button(L["run_diag"], use_container_width=True):
            if model and scaler:
                raw = np.array([list(u_vals.values())])
                is_safe = (model.predict(scaler.transform(raw))[0] == 1)
            else: is_safe = 6.5 <= u_vals["ph"] <= 8.5 and u_vals["Solids"] < 500
            
            # Origin Logic: If user uploads image or sensor values are high
            is_industrial = (u_vals['Solids'] > 600 or u_vals['Chloramines'] > 4 or diag_source_img is not None)
            source_key = "source_industrial" if is_industrial else "source_natural"
            
            st.session_state.last_result = {
                "safe": is_safe, "vals": u_vals, "status": L["verdict_safe"] if is_safe else L["verdict_unsafe"],
                "status_en": "Safe" if is_safe else "Unsafe", "place": place_name, "source": L[source_key], "source_en": LANG_DICT["English"][source_key]
            }
            st.session_state.history.append({
                "name": st.session_state.user_profile["name"], "age": st.session_state.user_profile["age"], "phone": st.session_state.user_profile["phone"], "address": st.session_state.user_profile["address"],
                "status": st.session_state.last_result["status"], "lat": lat, "lon": lon, "place": place_name, "time": datetime.datetime.now().strftime("%H:%M"), "feedback": f"Origin: {st.session_state.last_result['source']}"
            })

    with c2:
        m_diag = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker([lat, lon], icon=folium.Icon(color='blue')).add_to(m_diag)
        st_folium(m_diag, height=250, use_container_width=True, key="diag_map")
        
        st.subheader(L["who_head"])
        who_std = {"ph": (6.5, 8.5), "Hardness": 200, "Solids": 500, "Chloramines": 4, "Sulfate": 250, "Conductivity": 400, "Organic_carbon": 2, "Trihalomethanes": 0.8, "Turbidity": 5}
        rows = [{L["who_param"]: L["sensors"][k], L["who_limit"]: str(who_std[k]), L["who_result"]: "✅" if ((who_std[k][0] <= v <= who_std[k][1]) if k=="ph" else (v <= who_std[k])) else "❌"} for k, v in u_vals.items()]
        st.table(pd.DataFrame(rows))

    if st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown(f"<div style='background:{'#1b5e20' if res['safe'] else '#b71c1c'}; padding:20px; border-radius:10px; color:white; text-align:center;'><h2>{res['status']}</h2></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pollution-box'>🔍 {L['pollution_label']}: {res['source']}</div>", unsafe_allow_html=True)
        
        r1, r2 = st.columns(2)
        with r1:
            st.subheader(L["treatment_title"])
            st.markdown(f'<div class="treatment-card">{L["treatment_list"]}</div>', unsafe_allow_html=True)
        with r2:
            st.subheader(L["homemade_title"])
            st.markdown(f'<div class="home-card">{L["homemade_list"]}</div>', unsafe_allow_html=True)

        st.subheader(L["graph_title"])
        st.bar_chart(pd.DataFrame({"Levels": list(res['vals'].values())}, index=list(L['sensors'].values())))
        
        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "HYDROVIDEX AI REPORT", ln=True)
        pdf_bytes = pdf.output()
        st.download_button(L["download_report"], data=pdf_bytes, file_name="Report.pdf")

# --- TAB 2: VISUAL SCAN (ORIGIN + MICROPLASTIC) ---
with tabs[1]:
    st.header(L["tab2"])
    v_img = st.file_uploader(L["visual_explain_label"], type=['jpg','png','jpeg'], key="v_scan")
    if v_img:
        file_bytes = np.asarray(bytearray(v_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        conts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in conts if cv2.contourArea(c) > 5]
        
        # Calculation
        plastic_pct = (sum([cv2.contourArea(c) for c in valid]) / (img.shape[0]*img.shape[1])) * 100
        is_human_made = plastic_pct > 0.02 or len(valid) > 15 # Heuristic for human-made debris
        
        overlay = img_rgb.copy()
        cv2.drawContours(overlay, valid, -1, (0, 100, 255), 2)
        
        st.subheader("🖼️ Side-by-Side Analysis")
        v1, v2 = st.columns(2)
        v1.image(img_rgb, caption="Original Sample", use_container_width=True)
        v2.image(overlay, caption=f"Scanned: {plastic_pct:.4f}% Particles", use_container_width=True)
        
        # Origin Result & Solutions
        origin_text = L["source_industrial"] if is_human_made else L["source_natural"]
        color_box = "#b71c1c" if is_human_made else "#ef6c00"
        
        st.markdown(f"""
        <div class="analysis-card" style="border-left: 10px solid {color_box}">
            <h3>🔍 Origin Classification: {origin_text}</h3>
            <p><b>Microplastic Concentration:</b> {plastic_pct:.4f}%</p>
            <hr>
            <p><b>Recommended Action:</b> {'Requires Chemical Neutralization & Ultra-filtration' if is_human_made else 'Requires Boiling & Sedimentation'}</p>
            <p><b>Target Solutions:</b> {L['treatment_list'] if is_human_made else L['homemade_list']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🕒 Recent Scan Activity")
    for s in [{"n": "Arjun", "p": "CB Pur", "r": "Safe (Nature)"}, {"n": "Meera", "p": "CB Pur", "r": "Unsafe (Human-Made)"}]:
        st.markdown(f'<div class="comm-card"><b>📷 {s["n"]}</b> | 📍 {s["p"]} | <b>Result:</b> {s["r"]}</div>', unsafe_allow_html=True)

# --- TAB 3: COMMUNITY ---
with tabs[2]:
    st.header(L["comm_header"])
    m_c = folium.Map(location=[lat, lon], zoom_start=11)
    for h in st.session_state.history:
        folium.Marker([h.get('lat', lat), h.get('lon', lon)], popup=h.get('name')).add_to(m_c)
    st_folium(m_c, height=350, use_container_width=True, key="comm_map")
    for h in reversed(st.session_state.history): 
        st.markdown(f'<div class="comm-card"><b>👤 {h.get("name")}</b> ({h.get("age")}) | 📞 {h.get("phone")}<br>📍 {h.get("place")} | 💬 {h.get("feedback")}</div>', unsafe_allow_html=True)

# --- TAB 4: PROFILE ---
with tabs[3]:
    st.header(L["prof_head"])
    st.session_state.user_profile["name"] = st.text_input(L["name_label"], st.session_state.user_profile["name"])
    st.session_state.user_profile["phone"] = st.text_input(L["phone_label"], st.session_state.user_profile["phone"])
    st.session_state.user_profile["age"] = st.text_input(L["age_label"], st.session_state.user_profile["age"])
    st.session_state.user_profile["address"] = st.text_area(L["addr_label"], st.session_state.user_profile["address"])
    st.button(L["save_prof"])

st.caption("Hydrovidex AI Platform v27.0 | © 2026 Global Node")
