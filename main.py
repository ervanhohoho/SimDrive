# SimDrive Coach Prototype (Live Telemetry Version)
# Offline Assetto Corsa driving coach using Ollama, Streamlit, and live telemetry capture

import streamlit as st
import pandas as pd
import requests
import json
import time
import struct
import mmap
import os

# --- CONFIG ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # Change to your preferred Ollama model

# --- SHARED MEMORY CONFIG (Assetto Corsa) ---
AC_SHARED_MEMORY_NAME = "Local\\ACPMemoryMapFileName"
AC_GRAPHICS_MEMORY_NAME = "Local\\ACPMemoryMapFileGraphics"

st.set_page_config(page_title="SimDrive Coach Live", layout="wide")
st.title("🏎️ SimDrive Coach - Live Telemetry Mode")

st.write("This version connects directly to Assetto Corsa shared memory to capture live telemetry and provide AI feedback.")

# --- HELPER: Read shared memory ---
def read_shared_memory(name, size):
    try:
        file = mmap.mmap(-1, size, name)
        data = file.read(size)
        file.close()
        return data
    except Exception as e:
        return None

# --- CAPTURE TELEMETRY ---
def capture_telemetry(duration=10, interval=0.1):
    """Capture telemetry for N seconds."""
    telemetry_data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        # Assetto Corsa shared memory block sizes and layout simplified
        data = read_shared_memory(AC_SHARED_MEMORY_NAME, 2048)
        if not data:
            st.warning("Could not read shared memory. Make sure Assetto Corsa is running.")
            break
        # Example parsing (simplified mock, real struct parsing requires offsets)
        # This is a placeholder until we integrate the real AC telemetry SDK
        speed = struct.unpack_from('f', data, 0)[0] * 3.6  # Convert m/s to km/h
        throttle = struct.unpack_from('f', data, 4)[0]
        brake = struct.unpack_from('f', data, 8)[0]
        steering = struct.unpack_from('f', data, 12)[0]

        telemetry_data.append({
            'time': round(time.time() - start_time, 2),
            'speed': speed,
            'throttle': throttle,
            'brake': brake,
            'steering': steering
        })
        time.sleep(interval)

    return pd.DataFrame(telemetry_data)

# --- MAIN APP ---
mode = st.radio("Select Mode", ["Upload CSV", "Live Capture"], horizontal=True)

if mode == "Upload CSV":
    file = st.file_uploader("Upload your lap data (CSV)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.success("✅ Telemetry loaded!")
else:
    st.info("Click to capture live telemetry from Assetto Corsa (default 10s)")
    duration = st.slider("Capture Duration (seconds)", 5, 60, 10)
    if st.button("Start Capture"):
        with st.spinner("Capturing telemetry..."):
            df = capture_telemetry(duration=duration)
            st.success(f"✅ Captured {len(df)} samples!")

if 'df' in locals() and not df.empty:
    st.subheader("📊 Telemetry Overview")
    st.line_chart(df[['speed', 'throttle', 'brake']])

    summary = {
        "avg_speed": float(df['speed'].mean()),
        "max_speed": float(df['speed'].max()),
        "avg_throttle": float(df['throttle'].mean()),
        "avg_brake": float(df['brake'].mean()),
        "lap_time_est": float(len(df) / 60.0),
    }

    st.write("### 🧩 Summary:")
    st.json(summary)

    # --- Send to Ollama ---
    prompt = f"""
    You are a professional racing coach analyzing Assetto Corsa telemetry.
    Based on this summary: {json.dumps(summary)}
    Provide concise coaching advice on:
    - braking habits
    - throttle control
    - cornering technique
    - lap consistency
    End with a short motivational tip.
    """

    with st.spinner("Analyzing with Ollama model..."):
        try:
            response = requests.post(
                OLLAMA_API,
                json={"model": MODEL_NAME, "prompt": prompt},
                timeout=60
            )
            data = response.json()
            feedback = data.get("response", "No feedback received.")

            st.subheader("💬 AI Coaching Feedback")
            st.write(feedback)

        except Exception as e:
            st.error(f"❌ Error contacting Ollama: {e}")
