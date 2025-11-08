# SimDrive Coach Prototype (Session Capture + Auto Save/Load + Ollama Feedback)
# Offline Assetto Corsa driving coach using Ollama, Streamlit, and live session telemetry capture

import streamlit as st
import pandas as pd
import requests
import json
import time
import struct
import mmap
import threading
import os
from datetime import datetime

# --- CONFIG ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
SESSIONS_DIR = "sessions"

# --- ENSURE SESSIONS DIR ---
os.makedirs(SESSIONS_DIR, exist_ok=True)

# --- SHARED MEMORY CONFIG (Assetto Corsa) ---
AC_SHARED_MEMORY_NAME = "Local\\ACPMemoryMapFileName"
AC_GRAPHICS_MEMORY_NAME = "Local\\ACPMemoryMapFileGraphics"
AC_STATIC_MEMORY_NAME = "Local\\ACPMemoryMapFileStatic"

st.set_page_config(page_title="SimDrive Coach Live", layout="wide")
st.title("🏎️ SimDrive Coach - Live Session Capture")

st.write("Capture complete driving sessions from Assetto Corsa and get instant AI feedback with car and track info.")

# --- READ STATIC INFO (Car & Track) ---
def read_static_info():
    try:
        file = mmap.mmap(-1, 4096, AC_STATIC_MEMORY_NAME)
        data = file.read(4096)
        file.close()

        car_name = data[0:100].split(b'\x00', 1)[0].decode('utf-8')
        track_name = data[100:200].split(b'\x00', 1)[0].decode('utf-8')
        track_config = data[200:300].split(b'\x00', 1)[0].decode('utf-8')

        return {
            "car_name": car_name or "Unknown",
            "track_name": track_name or "Unknown",
            "track_config": track_config or "Unknown",
        }
    except Exception as e:
        return {"car_name": "Unknown", "track_name": "Unknown", "track_config": "Unknown", "error": str(e)}

# --- READ GRAPHICS DATA (Session State) ---
def read_graphics_info():
    try:
        file = mmap.mmap(-1, 2048, AC_GRAPHICS_MEMORY_NAME)
        data = file.read(2048)
        file.close()

        status = struct.unpack_from('i', data, 0)[0]
        session = struct.unpack_from('i', data, 4)[0]
        completed_laps = struct.unpack_from('i', data, 8)[0]

        return {"status": status, "session": session, "completed_laps": completed_laps}
    except Exception:
        return {"status": 0, "session": 0, "completed_laps": 0}

# --- READ PHYSICS DATA (Telemetry) ---
def read_physics_data():
    try:
        file = mmap.mmap(-1, 2048, AC_SHARED_MEMORY_NAME)
        data = file.read(2048)
        file.close()

        speed = struct.unpack_from('f', data, 0)[0] * 3.6
        throttle = struct.unpack_from('f', data, 4)[0]
        brake = struct.unpack_from('f', data, 8)[0]
        steering = struct.unpack_from('f', data, 12)[0]

        return {'speed': speed, 'throttle': throttle, 'brake': brake, 'steering': steering}
    except Exception:
        return None

# --- INITIALIZE SESSION STATE ---
if 'capturing' not in st.session_state:
    st.session_state.capturing = False
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []
if 'last_session_file' not in st.session_state:
    st.session_state.last_session_file = None

# --- CAPTURE THREAD ---
def capture_session():
    start_time = time.time()
    while st.session_state.capturing:
        physics = read_physics_data()
        graphics = read_graphics_info()
        if physics:
            row = {
                'time': round(time.time() - start_time, 2),
                'speed': physics['speed'],
                'throttle': physics['throttle'],
                'brake': physics['brake'],
                'steering': physics['steering'],
                'laps': graphics['completed_laps']
            }
            st.session_state.data_buffer.append(row)
        time.sleep(0.1)

# --- UI CONTROLS ---
static_info = read_static_info()
st.markdown(f"### 🏁 Car: **{static_info['car_name']}** | 🗺️ Track: **{static_info['track_name']} ({static_info['track_config']})**")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Session Capture", disabled=st.session_state.capturing):
        st.session_state.data_buffer.clear()
        st.session_state.capturing = True
        threading.Thread(target=capture_session, daemon=True).start()
        st.success("Capturing telemetry...")
with col2:
    if st.button("⏹️ Stop Capture", disabled=not st.session_state.capturing):
        st.session_state.capturing = False
        # --- AUTO SAVE ---
        if st.session_state.data_buffer:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{static_info['car_name'].replace(' ','_')}_{static_info['track_name'].replace(' ','_')}_{timestamp}.csv"
            filepath = os.path.join(SESSIONS_DIR, filename)
            pd.DataFrame(st.session_state.data_buffer).to_csv(filepath, index=False)
            st.session_state.last_session_file = filepath
            st.success(f"Session saved: {filepath}")

# --- AUTO LOAD LAST SESSION ---
if st.session_state.last_session_file and os.path.exists(st.session_state.last_session_file):
    df = pd.read_csv(st.session_state.last_session_file)
    st.subheader("📊 Last Captured Session Data")
    st.line_chart(df[['speed','throttle','brake']])

    summary = {
        "avg_speed": float(df['speed'].mean()),
        "max_speed": float(df['speed'].max()),
        "avg_throttle": float(df['throttle'].mean()),
        "avg_brake": float(df['brake'].mean()),
        "lap_count": int(df['laps'].max()),
        "session_length": float(df['time'].max()),
        "car": static_info['car_name'],
        "track": static_info['track_name'],
        "layout": static_info['track_config']
    }

    st.write("### 🧩 Summary:")
    st.json(summary)

    # --- SEND TO OLLAMA ---
    prompt = f"""
    You are a professional racing coach analyzing Assetto Corsa telemetry.
    Car: {summary['car']}
    Track: {summary['track']} ({summary['layout']})
    Session summary: {json.dumps(summary)}
    Give detailed yet concise coaching advice on:
    - braking technique
    - throttle control
    - cornering habits
    - consistency across laps
    - possible setup improvements for this car and track
    End with a short motivational tip.
    """

    with st.spinner("Analyzing with Ollama model..."):
        try:
            response = requests.post(
                OLLAMA_API,
                json={"model": MODEL_NAME, "prompt": prompt},
                timeout=90
            )
            data = response.json()
            feedback = data.get("response", "No feedback received.")

            st.subheader("💬 AI Coaching Feedback")
            st.write(feedback)

        except Exception as e:
            st.error(f"❌ Error contacting Ollama: {e}")
