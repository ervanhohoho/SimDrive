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
import altair as alt

# --- CONFIG ---
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
SESSIONS_DIR = "sessions"

# --- ENSURE SESSIONS DIR ---
os.makedirs(SESSIONS_DIR, exist_ok=True)

# --- SHARED MEMORY CONFIG (Assetto Corsa) ---
AC_SHARED_MEMORY_NAME = "Local\\acpmf_physics"
AC_GRAPHICS_MEMORY_NAME = "Local\\acpmf_graphics"
AC_STATIC_MEMORY_NAME = "Local\\acpmf_static"

st.set_page_config(page_title="SimDrive Coach Live", layout="wide")
st.title("🏎️ SimDrive Coach - Live Session Capture")

st.write("Capture complete driving sessions from Assetto Corsa and get instant AI feedback with car and track info.")

# --- READ STATIC INFO (Car & Track) ---
def read_static_info():
    try:
        file = mmap.mmap(-1, 4096, AC_STATIC_MEMORY_NAME)
        data = file.read(4096)
        file.close()

        # Assetto Corsa static memory structure may have version at offset 0
        # Try to find null-terminated strings starting from different offsets
        # Common structure: version (int) at 0, car name at 4, track at 104, config at 204
        # Or: car name at 0, track at 100, config at 200
        
        # Try reading from offset 0 first
        car_name_bytes = data[0:100]
        # If first 4 bytes look like an integer (version), skip them
        if len(car_name_bytes) >= 4:
            version_check = struct.unpack_from('i', car_name_bytes, 0)[0]
            # If it's a small integer (likely version), start from offset 4
            if 0 < version_check < 100:
                car_name_bytes = data[4:104]
                track_name_bytes = data[104:204]
                track_config_bytes = data[204:304]
            else:
                car_name_bytes = data[0:100]
                track_name_bytes = data[100:200]
                track_config_bytes = data[200:300]
        else:
            track_name_bytes = data[100:200]
            track_config_bytes = data[200:300]

        # Extract null-terminated strings
        car_name = car_name_bytes.split(b'\x00', 1)[0].decode('utf-8', errors='ignore').strip()
        track_name = track_name_bytes.split(b'\x00', 1)[0].decode('utf-8', errors='ignore').strip()
        track_config = track_config_bytes.split(b'\x00', 1)[0].decode('utf-8', errors='ignore').strip()

        # Clean up empty strings and invalid characters
        car_name = car_name if car_name and len(car_name) > 0 and not car_name.isdigit() else "Unknown"
        track_name = track_name if track_name and len(track_name) > 0 else "Unknown"
        track_config = track_config if track_config and len(track_config) > 0 else "Unknown"

        return {
            "car_name": car_name,
            "track_name": track_name,
            "track_config": track_config,
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

# --- HELPER FUNCTIONS ---
def get_session_files():
    """Get all session CSV files sorted by modification time (newest first)"""
    if not os.path.exists(SESSIONS_DIR):
        return []
    files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith('.csv')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SESSIONS_DIR, x)), reverse=True)
    return files

def get_analysis_file(csv_file):
    """Get the corresponding analysis JSON file for a CSV session file"""
    base_name = os.path.splitext(csv_file)[0]
    return os.path.join(SESSIONS_DIR, f"{base_name}_analysis.json")

def save_analysis(filepath, summary, feedback):
    """Save session analysis to JSON file"""
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "feedback": feedback
    }
    analysis_file = get_analysis_file(filepath)
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    return analysis_file

def load_analysis(filepath):
    """Load session analysis from JSON file"""
    analysis_file = get_analysis_file(filepath)
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# --- INITIALIZE SESSION STATE ---
if 'capturing' not in st.session_state:
    st.session_state.capturing = False
if 'capture_event' not in st.session_state:
    st.session_state.capture_event = threading.Event()
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []
if 'last_session_file' not in st.session_state:
    st.session_state.last_session_file = None
if 'selected_session' not in st.session_state:
    st.session_state.selected_session = None

# --- CAPTURE THREAD ---
def capture_session(capture_event, data_buffer):
    start_time = time.time()
    while capture_event.is_set():
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

# --- SIDEBAR: SESSION HISTORY ---
with st.sidebar:
    st.header("📚 Session History")
    session_files = get_session_files()
    
    if session_files:
        # Create a list of session names for selection
        session_options = ["Select a session..."] + session_files
        
        # Get the index of the currently selected session
        current_index = 0
        if st.session_state.selected_session:
            try:
                current_index = session_files.index(st.session_state.selected_session) + 1
            except ValueError:
                current_index = 0
        
        selected = st.selectbox(
            "Previous Sessions",
            options=session_options,
            index=current_index,
            key="session_selector"
        )
        
        if selected and selected != "Select a session...":
            st.session_state.selected_session = selected
        elif selected == "Select a session...":
            st.session_state.selected_session = None
        
        st.markdown("---")
        st.write(f"**Total Sessions:** {len(session_files)}")
    else:
        st.info("No previous sessions found.")
        st.session_state.selected_session = None

# --- UI CONTROLS ---
static_info = read_static_info()
st.markdown(f"### 🏁 Car: **{static_info['car_name']}** | 🗺️ Track: **{static_info['track_name']} ({static_info['track_config']})**")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Session Capture", disabled=st.session_state.capturing):
        st.session_state.data_buffer.clear()
        st.session_state.capturing = True
        st.session_state.capture_event.set()
        threading.Thread(target=capture_session, args=(st.session_state.capture_event, st.session_state.data_buffer), daemon=True).start()
        st.success("Capturing telemetry...")
with col2:
    if st.button("⏹️ Stop Capture", disabled=not st.session_state.capturing):
        st.session_state.capturing = False
        st.session_state.capture_event.clear()
        # --- AUTO SAVE ---
        if st.session_state.data_buffer:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{static_info['car_name'].replace(' ','_')}_{static_info['track_name'].replace(' ','_')}_{timestamp}.csv"
            filepath = os.path.join(SESSIONS_DIR, filename)
            pd.DataFrame(st.session_state.data_buffer).to_csv(filepath, index=False)
            st.session_state.last_session_file = filepath
            st.session_state.selected_session = filename
            st.success(f"Session saved: {filename}")

# --- LOAD SELECTED OR LAST SESSION ---
session_to_load = None

# Determine which session to load
if st.session_state.selected_session:
    session_path = os.path.join(SESSIONS_DIR, st.session_state.selected_session)
    if os.path.exists(session_path):
        session_to_load = session_path
elif st.session_state.last_session_file and os.path.exists(st.session_state.last_session_file):
    session_to_load = st.session_state.last_session_file

# --- DISPLAY SESSION DATA ---
if session_to_load:
    df = pd.read_csv(session_to_load)
    session_name = os.path.basename(session_to_load)
    st.subheader(f"📊 Session Data: {session_name}")
    
    # Create separate charts for speed and throttle/brake
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Speed (km/h)**")
        # Format speed values for better readability - round to nearest integer
        df_chart = df.copy()
        df_chart['speed_rounded'] = df_chart['speed'].round(0)
        speed_chart = alt.Chart(df_chart.reset_index()).mark_line(strokeWidth=2).encode(
            x=alt.X('index:Q', title='Time (samples)'),
            y=alt.Y('speed_rounded:Q', title='Speed (km/h)', 
                    axis=alt.Axis(format='.0f')),
            color=alt.value('#1f77b4')
        ).properties(width=350, height=300)
        st.altair_chart(speed_chart, use_container_width=True)
    
    with col2:
        st.write("**Throttle & Brake**")
        controls_data = df[['throttle','brake']].copy().reset_index().melt(id_vars='index', var_name='metric', value_name='value')
        controls_chart = alt.Chart(controls_data).mark_line(strokeWidth=2).encode(
            x=alt.X('index:Q', title='Time (samples)'),
            y=alt.Y('value:Q', scale=alt.Scale(domain=[0, 1]), title='Value'),
            color='metric:N'
        ).properties(width=350, height=300)
        st.altair_chart(controls_chart, use_container_width=True)

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

    st.write("### 🧩 Session Summary:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏁 Car", summary['car'])
        st.metric("🗺️ Track", f"{summary['track']} ({summary['layout']})")
        st.metric("⏱️ Session Length", f"{summary['session_length']:.1f} seconds")
    with col2:
        st.metric("📊 Average Speed", f"{summary['avg_speed']:.0f} km/h")
        st.metric("⚡ Max Speed", f"{summary['max_speed']:.0f} km/h")
        st.metric("🏁 Laps Completed", summary['lap_count'])
    with col3:
        st.metric("🚗 Average Throttle", f"{summary['avg_throttle']:.1%}")
        st.metric("🛑 Average Brake", f"{summary['avg_brake']:.1%}")

    # --- AI COACHING FEEDBACK ---
    existing_analysis = load_analysis(session_to_load)
    
    # Check if we should generate new analysis
    generate_analysis = False
    if not existing_analysis:
        generate_analysis = True
    else:
        # Add button to regenerate analysis
        if st.button("🔄 Regenerate AI Analysis"):
            generate_analysis = True
    
    if generate_analysis:
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
                    json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                    timeout=90
                )
                response.raise_for_status()
                data = response.json()
                
                # Ollama API returns "response" field for non-streaming, or we need to handle streaming
                feedback = data.get("response", "")
                
                # If response is empty, try to get it from the data structure
                if not feedback and isinstance(data, dict):
                    # Sometimes the response might be in a different format
                    feedback = str(data).strip()
                
                if not feedback:
                    feedback = "No feedback received from Ollama. Please check if Ollama is running and the model is available."

                # Save analysis to JSON file
                try:
                    analysis_file = save_analysis(session_to_load, summary, feedback)
                    existing_analysis = load_analysis(session_to_load)  # Reload after saving
                    st.success(f"✅ Analysis saved to: {os.path.basename(analysis_file)}")
                except Exception as e:
                    st.warning(f"⚠️ Could not save analysis: {e}")
                    # Still create analysis object for display even if save failed
                    existing_analysis = {
                        "feedback": feedback,
                        "timestamp": datetime.now().isoformat()
                    }

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error contacting Ollama: {e}")
                st.info("Make sure Ollama is running on localhost:11434 and the model is available.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
    
    # Display analysis (either existing or newly generated)
    if existing_analysis:
        st.subheader("💬 AI Coaching Feedback")
        st.markdown(existing_analysis.get("feedback", "No feedback available."))
        st.caption(f"Analysis saved on: {existing_analysis.get('timestamp', 'Unknown')}")
