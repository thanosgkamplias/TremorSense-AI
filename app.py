import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import time
from collections import deque

# --- ΠΡΟΣΤΑΣΙΑ ΓΙΑ ΤΟ CLOUD DEPLOYMENT ---
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(page_title="TremorSense AI", layout="centered", page_icon="🧠")

# Custom CSS for a modern, clinical, and minimal UI
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: transparent; 
        color: #1E88E5;                     
        border: 1.5px solid #1E88E5;        
        border-radius: 6px;                 
        padding: 6px 16px;                  
        font-size: 14px;                    
        font-weight: 500;
        box-shadow: none;                   
        transition: all 0.2s ease;
        display: block;
        margin: 0 auto;                     
    }
    div.stButton > button:first-child:hover {
        background-color: rgba(30, 136, 229, 0.08); 
        color: #1565C0;
        border-color: #1565C0;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. HEADER SECTION ---
try:
    st.image("tremorsense_logo.png", width=300)
except:
    st.title("🧠 TremorSense AI") # Fallback if logo is missing
st.markdown("#### Real-time Neurological Tremor Analysis System")

st.markdown("""
    <p style='font-size: 16px; color: #808495; line-height: 1.5;'>
        An AI-powered application that analyzes motion data from wearable sensors to instantly 
        detect and differentiate between Parkinson's Disease, Essential Tremor, and healthy movements.
    </p>
""", unsafe_allow_html=True)
st.divider()

# --- 3. SIDEBAR: MODE SELECTION ---
st.sidebar.header("⚙️ System Settings")

mode = st.sidebar.radio(
    "Data Source Mode:",
    ("☁️ Cloud Demo (Playback)", "🔌 Live Hardware (USB)")
)

selected_port = None
demo_class = None

if mode == "🔌 Live Hardware (USB)":
    if not HAS_SERIAL:
        st.sidebar.info("💡 **Edge IoT Mode:** Live USB streaming is designed for local environments. Switch to **Cloud Demo**.")
    else:
        available_ports = [port.device for port in serial.tools.list_ports.comports()]
        if not available_ports:
            # ΤΟ ΝΕΟ ΕΠΑΓΓΕΛΜΑΤΙΚΟ ΜΗΝΥΜΑ
            st.sidebar.info(
                "💡 **Edge IoT Mode**\n\n"
                "No hardware detected.\n\n"
                "👉 **Viewing online?** Switch to **Cloud Demo (Playback)** above to see the AI in action.\n\n"
                "👉 **Running locally?** Plug in your Arduino Nano 33 BLE via USB."
            )
        else:
            selected_port = st.sidebar.selectbox("Select Device Port", available_ports)
        
elif mode == "☁️ Cloud Demo (Playback)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("☁️ Playback Options")
    demo_class = st.sidebar.selectbox(
        "Select Pathology to Simulate:",
        ("Normal (Healthy Baseline)", "Parkinson's Disease", "Essential Tremor")
    )

st.sidebar.markdown("---")
st.sidebar.info("Model Accuracy: **92.2%**\n\nSensor Target: **100 Hz**")

# --- 4. LOAD AI MODEL ---
@st.cache_resource
def load_ai_model():
    return load_model('tremor_model.keras')

try:
    model = load_ai_model()
    st.toast("AI Model loaded successfully!", icon="✅")
except Exception as e:
    st.error("Failed to load 'tremor_model.keras'. Please ensure the file is in the directory.")
    st.stop()

# Model Output Classes mapping
classes = {
    0: ("🟠 ESSENTIAL TREMOR (Action)", "warning"),
    1: ("✅ NORMAL (No Tremor)", "success"),
    2: ("⚠️ PARKINSON'S (Resting)", "error")
}

# --- 5. UI PLACEHOLDERS ---
status_placeholder = st.empty()
chart_placeholder = st.empty()
confidence_title = st.empty() 

col1, col2, col3 = st.columns(3)
prog_ess = col1.empty()
prog_norm = col2.empty()
prog_park = col3.empty()

# --- 6. CORE UPDATE FUNCTION (ME SMOOTHING) ---
def update_dashboard(window_data, pred_buffer):
    """ window_data is a numpy array of shape (100, 9) """
    
    # 1. DATA PREPROCESSING
    window_centered = window_data - np.mean(window_data, axis=0)
    input_data = window_centered.reshape(1, 100, 9)
    
    # 2. RAW AI PREDICTION
    raw_predictions = model.predict(input_data, verbose=0)[0]
    
    # 3. SMOOTHING
    pred_buffer.append(raw_predictions)
    smoothed_predictions = np.mean(pred_buffer, axis=0)
    
    winner_index = np.argmax(smoothed_predictions)
    winner_name, status_type = classes[winner_index]
    
    # 4. UPDATE UI STATUS
    if status_type == "success":
        status_placeholder.success(f"DIAGNOSIS: **{winner_name}**")
    elif status_type == "error":
        status_placeholder.error(f"DIAGNOSIS: **{winner_name}**")
    else:
        status_placeholder.warning(f"DIAGNOSIS: **{winner_name}**")
        
    # 5. UPDATE CHART
    df_chart = pd.DataFrame(window_data, columns=['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ'])
    df_chart['Tremor_Magnitude'] = np.sqrt(df_chart['accX']**2 + df_chart['accY']**2 + df_chart['accZ']**2)
    chart_placeholder.line_chart(df_chart['Tremor_Magnitude'], height=250)
    
    # 6. UPDATE PROGRESS BARS
    prog_ess.progress(float(smoothed_predictions[0]), text=f"Essential: {smoothed_predictions[0]*100:.1f}%")
    prog_norm.progress(float(smoothed_predictions[1]), text=f"Normal: {smoothed_predictions[1]*100:.1f}%")
    prog_park.progress(float(smoothed_predictions[2]), text=f"Parkinson's: {smoothed_predictions[2]*100:.1f}%")


# --- 7. THE MAIN EXECUTION LOOP ---
button_text = "Start Live Diagnosis" if mode == "🔌 Live Hardware (USB)" else f"Start {demo_class} Simulation"

if st.button(button_text, use_container_width=False):
    confidence_title.write("**Real-time AI Confidence:**")
    
    prediction_buffer = deque(maxlen=5) 
    
# ==========================================
    # MODE A: LIVE USB STREAMING
    # ==========================================
    if mode == "🔌 Live Hardware (USB)":
        if not HAS_SERIAL or not selected_port:
            # ΝΕΟ ΜΗΝΥΜΑ ΑΝΤΙ ΓΙΑ ERROR
            st.warning("⚠️ **Hardware Required:** To run real-time inference, an Arduino must be connected via USB. Please switch to the **☁️ Cloud Demo (Playback)** mode from the sidebar to evaluate the AI's performance.", icon="🤖")
            st.stop()

        try:
            ser = serial.Serial(selected_port, 115200, timeout=1)
            st.toast(f"Connected to {selected_port}", icon="🔌")
            buffer = []
            
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    values = [float(x) for x in line.split(',')]
                    if len(values) == 9:
                        buffer.append(values)
                
                if len(buffer) == 100:
                    window_array = np.array(buffer)
                    update_dashboard(window_array, prediction_buffer) 
                    buffer = [] 

        except serial.SerialException:
            st.error(f"Connection failed! Make sure {selected_port} is not in use by the Arduino IDE.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # ==========================================
    # MODE B: CLOUD DEMO (PLAYBACK) - 1 MINUTE DURATION
    # ==========================================
    elif mode == "☁️ Cloud Demo (Playback)":
        st.toast(f"Starting 60-second Simulation for {demo_class}...", icon="☁️")
        
        try:
            df_demo = pd.read_csv('tremor_dataset.csv')
            
            label_mapping = {
                "Normal (Healthy Baseline)": "Normal",
                "Parkinson's Disease": "Parkinson_Tremor",
                "Essential Tremor": "Essential_Tremor"
            }
            target_label = label_mapping[demo_class]
            
            filtered_df = df_demo[df_demo['Label'] == target_label]
            
            if filtered_df.empty:
                st.error(f"No data found for {demo_class}!")
                st.stop()
                
            sensor_data = filtered_df[['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ']].values
            
            window_size = 100
            step = 25 
            
            duration_seconds = 60
            start_time = time.time()
            current_index = 0
            
            while time.time() - start_time < duration_seconds:
                if current_index + window_size > len(sensor_data):
                    current_index = 0
                
                window_array = sensor_data[current_index : current_index + window_size, :]
                
                update_dashboard(window_array, prediction_buffer) 
                
                current_index += step
                time.sleep(0.12) 
                
            st.success(f"1-Minute Continuous Simulation for {demo_class} completed successfully!")
            
        except FileNotFoundError:
            st.error("❌ Dataset not found! Make sure 'tremor_dataset.csv' exists.")
