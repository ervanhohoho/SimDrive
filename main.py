# SimDrive Coach Prototype
# Offline Assetto Corsa driving coach using Ollama and Streamlit

import streamlit as st
import pandas as pd
import requests
import json

# --- CONFIG ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"  # You can change to mistral, phi3, etc.

st.set_page_config(page_title="SimDrive Coach", layout="wide")
st.title("🏎️ SimDrive Coach (Prototype)")

st.write("This is an offline driving coach powered by Ollama. Upload your Assetto Corsa telemetry data (CSV) to get AI feedback.")

# --- FILE UPLOAD ---
file = st.file_uploader("Upload your lap data (CSV)", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.success("✅ Telemetry loaded!")

    st.subheader("📊 Basic Telemetry Overview")
    st.write(df.describe())

    if {'speed', 'throttle', 'brake'}.issubset(df.columns):
        st.line_chart(df[['speed', 'throttle', 'brake']])

        # --- Summarize telemetry ---
        summary = {
            "avg_speed": float(df['speed'].mean()),
            "max_speed": float(df['speed'].max()),
            "avg_throttle": float(df['throttle'].mean()),
            "avg_brake": float(df['brake'].mean()),
            "lap_time_est": float(len(df) / 60.0),  # rough estimation
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

    else:
        st.warning("CSV must contain columns: speed, throttle, brake")
else:
    st.info("Upload a CSV file to start.")
