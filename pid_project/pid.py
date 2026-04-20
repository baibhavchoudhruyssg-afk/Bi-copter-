import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import random

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Bicopter Bench", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;900&family=Syne:wght@300;400;700&display=swap');
:root { --cyan:#00f5ff; --purple:#bf00ff; --red:#f38ba8; --green:#a6e3a1; --bg:#010a14; }
html,body,[class*="css"]{font-family:'Syne',sans-serif;background-color:var(--bg);color:#cdd6f4;}
.stApp{ background: radial-gradient(ellipse 80% 50% at 10% 0%,rgba(0,245,255,0.05) 0%,transparent 60%), #010a14; }
.stSidebar{background:rgba(2,12,24,0.97)!important;}
.stMetric{background:rgba(255,255,255,0.03);border:1px solid rgba(0,245,255,0.1);border-radius:6px;padding:10px!important;}
[data-testid="stMetricValue"]{color:var(--cyan)!important;font-family:'Orbitron'!important;font-size:1.2rem!important;}
[data-testid="stMetricLabel"]{color:rgba(0,245,255,0.6)!important;font-family:'Share Tech Mono'!important;font-size:0.8rem!important;}
.title{font-family:'Orbitron',sans-serif;font-size:2.5rem;font-weight:900;color:var(--cyan);text-shadow:0 0 20px rgba(0,245,255,0.4);margin-bottom:10px;}
.desc-box{background:rgba(255,255,255,0.05);border-left:3px solid var(--purple);padding:15px;border-radius:5px;margin-bottom:20px;font-size:0.95rem;color:#a6adc8;}
.section-title {font-family:'Share Tech Mono'; color:#a6adc8; font-size:1rem; letter-spacing:2px; margin-top:20px; margin-bottom:5px; border-bottom: 1px solid rgba(0,245,255,0.2); padding-bottom:5px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PHYSICS STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════
if 'true_angle' not in st.session_state:
    st.session_state.update({
        'true_angle': 0.0, 'velocity': 0.0, 
        'integral': 0.0, 'prev_error': 0.0,
        'est_angle': 0.0, 'est_error': 1.0,
        'raw_sensor': 0.0, 'measured_angle': 0.0,
        'motor_left': 1500, 'motor_right': 1500,
        'start_time': time.time(),
        'hist_time': [], 'hist_angle': [], 'hist_target': []
    })

# ══════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h3 style='font-family:Orbitron;color:#00f5ff;'>🎛 CONTROL PANEL</h3>", unsafe_allow_html=True)
    
    run_sim = st.toggle("🚀 RUN LIVE SIMULATION", value=False)
    
    if st.button("🌪️ KICK BICOPTER (Disturbance)", use_container_width=True):
        st.session_state.velocity += random.choice([-150.0, 150.0])

    st.markdown("<hr style='border:1px solid rgba(0,245,255,0.1);'>", unsafe_allow_html=True)

    target_angle = st.slider("🎯 Target Angle (deg)", -45.0, 45.0, 0.0, 1.0)
    
    st.markdown("### PID Tuning")
    kp = st.slider("Proportional (P)", 0.0, 10.0, 2.5, 0.1)
    ki = st.slider("Integral (I)", 0.0, 1.0, 0.05, 0.01)
    kd = st.slider("Derivative (D)", 0.0, 10.0, 1.5, 0.1)

    st.markdown("### Filter Settings")
    use_kalman = st.toggle("🟢 Enable Kalman Filter", value=False)
    noise_lvl = st.slider("Sensor Noise (Jitter)", 0.0, 10.0, 5.0, 0.5)
    
    if st.button("🔄 Reset Physics"):
        st.session_state.true_angle = 0.0
        st.session_state.velocity = 0.0
        st.session_state.integral = 0.0
        st.session_state.hist_time.clear()
        st.session_state.hist_angle.clear()
        st.session_state.hist_target.clear()
        st.session_state.start_time = time.time()

# ══════════════════════════════════════════════════════════════════
# PHYSICS ENGINE TICK
# ══════════════════════════════════════════════════════════════════
dt = 0.05  

if run_sim:
    st.session_state.raw_sensor = st.session_state.true_angle + random.uniform(-noise_lvl, noise_lvl)

    # Kalman Filter Math (1D)
    process_noise = 0.1
    sensor_noise_variance = 2.0
    
    est_pred = st.session_state.est_angle + (st.session_state.velocity * dt)
    err_pred = st.session_state.est_error + process_noise
    
    k_gain = err_pred / (err_pred + sensor_noise_variance)
    st.session_state.est_angle = est_pred + k_gain * (st.session_state.raw_sensor - est_pred)
    st.session_state.est_error = (1 - k_gain) * err_pred

    measured = st.session_state.est_angle if use_kalman else st.session_state.raw_sensor
    st.session_state.measured_angle = measured

    # PID Loop
    error = target_angle - measured
    st.session_state.integral = max(min(st.session_state.integral + error * dt, 20), -20)
    derivative = (error - st.session_state.prev_error) / dt
    st.session_state.prev_error = error

    output = (kp * error) + (ki * st.session_state.integral) + (kd * derivative)

    # Motor Mixing
    st.session_state.motor_left = max(1000, min(2000, 1500 - output * 10))
    st.session_state.motor_right = max(1000, min(2000, 1500 + output * 10))

    # Apply Torque & Physics
    torque = (st.session_state.motor_left - st.session_state.motor_right) * 0.005
    st.session_state.velocity += torque * dt
    st.session_state.velocity *= 0.95  
    st.session_state.true_angle += st.session_state.velocity * dt

    # 🛑 THE PHYSICAL STOPS (Lock at 60 and -60)
    if st.session_state.true_angle > 60.0:
        st.session_state.true_angle = 60.0
        st.session_state.velocity = 0.0  # Crash stop! Lose momentum.
    elif st.session_state.true_angle < -60.0:
        st.session_state.true_angle = -60.0
        st.session_state.velocity = 0.0  # Crash stop! Lose momentum.

    # Record History
    current_time = time.time() - st.session_state.start_time
    st.session_state.hist_time.append(current_time)
    st.session_state.hist_angle.append(st.session_state.true_angle)
    st.session_state.hist_target.append(target_angle)

    if len(st.session_state.hist_time) > 80:
        st.session_state.hist_time.pop(0)
        st.session_state.hist_angle.pop(0)
        st.session_state.hist_target.pop(0)

# ══════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='title'>Bicopter Single-Axis Bench</div>", unsafe_allow_html=True)

# Diagnostics Row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("True Angle", f"{st.session_state.true_angle:.1f}°")
m2.metric("Raw Sensor", f"{st.session_state.raw_sensor:.1f}°")
m3.metric("Filtered (Kalman)", f"{st.session_state.est_angle:.1f}°")
m4.metric("Left Motor", f"{int(st.session_state.motor_left)} PWM", delta_color="off")
m5.metric("Right Motor", f"{int(st.session_state.motor_right)} PWM", delta_color="off")

# ══════════════════════════════════════════════════════════════════
# REAL-TIME PID GRAPH
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>📡 REAL-TIME STABILIZATION TELEMETRY</div>", unsafe_allow_html=True)

fig_wave = go.Figure()

if len(st.session_state.hist_time) > 0:
    fig_wave.add_trace(go.Scatter(
        x=st.session_state.hist_time, y=st.session_state.hist_target, 
        mode='lines', name='Target Setpoint', line=dict(color='#bf00ff', width=2, dash='dot')
    ))
    fig_wave.add_trace(go.Scatter(
        x=st.session_state.hist_time, y=st.session_state.hist_angle, 
        mode='lines', name='Actual Angle', line=dict(color='#00f5ff', width=3),
        fill='tozeroy', fillcolor='rgba(0,245,255,0.05)'
    ))

fig_wave.update_layout(
    uirevision='constant', # 🚨 This prevents the graph from flashing!
    height=250, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Time (s)", gridcolor='rgba(0,245,255,0.1)', showticklabels=False),
    yaxis=dict(title="Angle (deg)", gridcolor='rgba(0,245,255,0.1)', range=[-90, 90]), 
    legend=dict(x=0.01, y=0.99, font=dict(color='white'))
)
st.plotly_chart(fig_wave, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# 3D VISUALIZATION
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>🚁 LIVE 3D KINEMATICS</div>", unsafe_allow_html=True)

arm_len = 5
rad = np.radians(st.session_state.true_angle)
dx = arm_len * np.cos(rad)
dy = arm_len * np.sin(rad) 

fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-5, 0], mode='lines', 
                           line=dict(color='#45475a', width=15), showlegend=False, hoverinfo='skip'))
fig_3d.add_trace(go.Scatter3d(x=[-dx, dx], y=[0, 0], z=[-dy, dy], mode='lines', 
                           line=dict(color='#89b4fa', width=10), showlegend=False, hoverinfo='skip'))
fig_3d.add_trace(go.Scatter3d(x=[-dx, dx], y=[0, 0], z=[-dy, dy], mode='markers', 
                           marker=dict(size=20, color=['#f38ba8', '#a6e3a1'], line=dict(color='white', width=2)),
                           name='Motors', hoverinfo='skip'))

t_rad = np.radians(target_angle)
fig_3d.add_trace(go.Scatter3d(x=[-arm_len*np.cos(t_rad), arm_len*np.cos(t_rad)], y=[0,0], 
                           z=[-arm_len*np.sin(t_rad), arm_len*np.sin(t_rad)], 
                           mode='lines', line=dict(color='rgba(255,255,255,0.2)', width=3, dash='dot'), 
                           name='Target Line', hoverinfo='skip'))

fig_3d.update_layout(
    uirevision='constant', # 🚨 This prevents the 3D model from flashing!
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0), height=450,
    scene=dict(
        xaxis=dict(visible=False, range=[-6, 6]),
        yaxis=dict(visible=False, range=[-3, 3]),
        zaxis=dict(visible=False, range=[-6, 6]),
        camera=dict(eye=dict(x=0, y=-2.5, z=0.5)) 
    )
)
st.plotly_chart(fig_3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# THE RERUN LOOP
# ══════════════════════════════════════════════════════════════════
if run_sim:
    time.sleep(dt)
    st.rerun()