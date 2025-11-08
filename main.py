# SimDrive Coach Prototype (Session Capture + Car/Track Detection)
# Offline Assetto Corsa driving coach using Ollama, Streamlit, and live session telemetry capture

import streamlit as st
import pandas as pd
import requests
import json
import time
import struct
import mmap
import threading

# --- CONFIG ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"

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

        # Parse basic session state (status and completed laps)
        status = struct.unpack_from('i', data, 0)[0]  # 1=menu, 2=replay, 3=live
        session = struct.unpack_from('i', data, 4)[0]  # 0=practice,1=qualify,2=race
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

        speed = struct.unpack_from('f', data, 0)[0] * 3.6  # m/s to km/h
        throttle = struct.unpack_from('f', data, 4)[0]
        brake = struct.unpack_from('f', data, 8)[0]
        steering = struct.unpack_from('f', data, 12)[0]

        return {
            'speed': speed,
            'throttle': throttle,
            'brake': brake,
            'steering': steering
        }
    except Exception:
        return None

# --- CAPTURE THREAD ---
capturing = False
data_buffer = []

def capture_session():
    global capturing, data_buffer
    start_time = time.time()
    while capturing:
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
            data_buffer.append(row)
        time.sleep(0.1)

# --- UI CONTROLS ---
static_info = read_static_info()
st.markdown(f"### 🏁 Car: **{static_info['car_name']}** | 🗺️ Track: **{static_info['track_name']} ({static_info['track_config']})**")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Session Capture"):
        data_buffer.clear()
        capturing = True
        thread = threading.Thread(target=capture_session)
        thread.start()
        st.success("Capturing telemetry...")
with col2:
    if st.button("⏹️ Stop Capture"):
        capturing = False
        st.success("Capture stopped.")

# --- DISPLAY & ANALYSIS ---
if not capturing and data_buffer:
    df = pd.DataFrame(data_buffer)
    st.subheader("📊 Captured Session Data")
    st.line_chart(df[['speed', 'throttle', 'brake']])

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

    # --- Send to Ollama ---
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
