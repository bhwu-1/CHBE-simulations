import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PID Flow Rate Controller", layout="wide")

st.title("⚙️ PID Flow Rate Controller Simulator")
st.markdown(
    "Simulate a **flow rate control loop** in a pipe system. "
    "Adjust the PID gains and see how the controller responds to a setpoint change."
)

# ── Sidebar: PID tuning + process parameters ─────────────────────────────────
st.sidebar.header("🎛️ PID Tuning Parameters")

Kp = st.sidebar.slider("Proportional Gain (Kp)", 0.1, 10.0, 1.5, 0.1,
                        help="How aggressively the controller reacts to current error.")
Ki = st.sidebar.slider("Integral Gain (Ki)", 0.0, 5.0, 0.5, 0.05,
                        help="Eliminates steady-state offset by accumulating past error.")
Kd = st.sidebar.slider("Derivative Gain (Kd)", 0.0, 2.0, 0.1, 0.05,
                        help="Dampens overshoot by reacting to rate of error change.")

st.sidebar.markdown("---")
st.sidebar.header("🏭 Process Parameters")

setpoint   = st.sidebar.slider("Setpoint Flow Rate (L/min)", 10.0, 100.0, 50.0, 1.0)
tau_p      = st.sidebar.slider("Process Time Constant τ (s)", 1.0, 20.0, 5.0, 0.5,
                                help="How slowly the physical pipe+valve system responds.")
K_process  = st.sidebar.slider("Process Gain Kₚ (L/min per %open)", 0.5, 3.0, 1.0, 0.1,
                                help="Flow change per unit valve opening.")
dead_time  = st.sidebar.slider("Dead Time θ (s)", 0.0, 5.0, 1.0, 0.1,
                                help="Measurement/transport delay before controller sees the response.")

st.sidebar.markdown("---")
st.sidebar.header("⚡ Disturbance")
add_dist   = st.sidebar.checkbox("Add inlet pressure disturbance", value=False)
dist_time  = st.sidebar.slider("Disturbance at t= (s)", 10, 80, 40, 1)
dist_mag   = st.sidebar.slider("Disturbance magnitude (L/min)", -20.0, 20.0, -10.0, 1.0)

# ── Simulation ────────────────────────────────────────────────────────────────
dt      = 0.1          # time step (s)
t_end   = 100.0
t       = np.arange(0, t_end, dt)
n       = len(t)

# Valve output limits (% open)
u_min, u_max = 0.0, 100.0

flow       = np.zeros(n)   # measured flow rate
valve      = np.zeros(n)   # valve position (%)
error      = np.zeros(n)
integral   = np.zeros(n)
derivative = np.zeros(n)

# Dead-time buffer (steps)
delay_steps = max(1, int(dead_time / dt))
flow_buffer = np.zeros(delay_steps)

# Initial conditions — start at 0, setpoint kicks in at t=0
flow[0]  = 0.0
valve[0] = 50.0

integrator = 0.0
prev_error = setpoint - flow[0]

# First-order process state
x_process = flow[0]   # internal process state

for i in range(1, n):
    # Disturbance
    dist = dist_mag if (add_dist and t[i] >= dist_time) else 0.0

    # PID error (using delayed measurement)
    measured_flow = flow_buffer[-1]
    e = setpoint - measured_flow
    error[i] = e

    # Integral (with anti-windup clamping)
    integrator += e * dt
    integrator  = np.clip(integrator, -200, 200)
    integral[i] = integrator

    # Derivative
    d = (e - prev_error) / dt
    derivative[i] = d
    prev_error = e

    # Controller output
    u = Kp * e + Ki * integrator + Kd * d
    u = np.clip(u, u_min, u_max)
    valve[i] = u

    # First-order process: τ dx/dt = -x + Kp*u + disturbance
    dxdt = (-x_process + K_process * u + dist) / tau_p
    x_process += dxdt * dt
    x_process  = max(x_process, 0.0)

    # Update dead-time buffer (shift & insert new value)
    flow_buffer = np.roll(flow_buffer, 1)
    flow_buffer[0] = x_process
    flow[i] = x_process

# ── Performance metrics ───────────────────────────────────────────────────────
steady_state_start = int(0.9 * n)
ss_flow  = np.mean(flow[steady_state_start:])
ss_error = abs(setpoint - ss_flow)

# Rise time: time to first reach 90% of setpoint
rise_idx = np.where(flow >= 0.9 * setpoint)[0]
rise_time = t[rise_idx[0]] if len(rise_idx) > 0 else float("nan")

# Overshoot
overshoot = max(0, (np.max(flow) - setpoint) / setpoint * 100)

# Settling time: last time |error| > 2% of setpoint
tol = 0.02 * setpoint
outside = np.where(np.abs(flow - setpoint) > tol)[0]
settling_time = t[outside[-1]] if len(outside) > 0 else 0.0

# ── Plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9), facecolor="#0e1117")
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ACCENT   = "#00c8ff"
GREEN    = "#00e676"
ORANGE   = "#ff9800"
RED      = "#ff4444"
BG       = "#0e1117"
PANEL_BG = "#1a1d27"
TEXT     = "#e0e0e0"

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color="#2a2d3a", linewidth=0.5, linestyle="--")

# 1 — Flow rate response
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, flow,                color=ACCENT,  lw=2,   label="Measured Flow")
ax1.axhline(setpoint,            color=GREEN,   lw=1.5, linestyle="--", label=f"Setpoint ({setpoint:.0f} L/min)")
ax1.axhline(setpoint * 1.02,     color=RED,     lw=0.8, linestyle=":",  alpha=0.6, label="±2% band")
ax1.axhline(setpoint * 0.98,     color=RED,     lw=0.8, linestyle=":",  alpha=0.6)
if add_dist:
    ax1.axvline(dist_time, color=ORANGE, lw=1.2, linestyle="--", alpha=0.7, label="Disturbance")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Flow Rate (L/min)")
ax1.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=9)
style_ax(ax1, "Flow Rate Response")

# 2 — Valve position
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, valve, color=ORANGE, lw=1.8)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Valve Opening (%)")
style_ax(ax2, "Controller Output (Valve Position)")

# 3 — Error
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, error, color=RED, lw=1.5)
ax3.axhline(0, color="#888", lw=0.8, linestyle="--")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error (L/min)")
style_ax(ax3, "Control Error  (Setpoint − Measured)")

# 4 — Integral term
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t, Ki * integral, color=GREEN, lw=1.5)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Ki · ∫e dt  (L/min)")
style_ax(ax4, "Integral Term Contribution")

# 5 — Derivative term
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(t, Kd * derivative, color="#bb86fc", lw=1.5)
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Kd · de/dt  (L/min)")
style_ax(ax5, "Derivative Term Contribution")

fig.patch.set_facecolor(BG)
st.pyplot(fig)
plt.close(fig)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rise Time",        f"{rise_time:.1f} s"   if not np.isnan(rise_time) else "N/A")
c2.metric("Settling Time",    f"{settling_time:.1f} s")
c3.metric("Overshoot",        f"{overshoot:.1f} %")
c4.metric("Steady-State Error", f"{ss_error:.2f} L/min")

# ── Engineering explanation ───────────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Engineering Background", expanded=False):
    st.markdown("""
    ### Process Model
    The flow rate through a pipe is modelled as a **first-order system** driven by a control valve:

    $$\\tau \\frac{dF}{dt} = -F + K_p \\cdot u + d(t)$$

    | Symbol | Meaning |
    |--------|---------|
    | $F$ | Flow rate (L/min) |
    | $\\tau$ | Process time constant — how quickly flow responds to valve change |
    | $K_p$ | Process gain — L/min per % valve opening |
    | $u$ | Valve opening (0–100 %) |
    | $d(t)$ | External disturbance (e.g. inlet pressure drop) |

    ### PID Controller
    $$u(t) = K_p \\, e(t) + K_i \\int_0^t e(\\tau)\\,d\\tau + K_d \\frac{de}{dt}$$

    - **Proportional (Kp):** Immediate reaction to error. High Kp = faster response but more overshoot.
    - **Integral (Ki):** Drives steady-state error to zero over time. Too high = oscillation.
    - **Derivative (Kd):** Anticipates error trend, dampens overshoot. Too high = noise amplification.

    ### Dead Time
    Represents the transport delay — the time it takes for fluid to travel from the valve to the flow sensor.
    Dead time makes the system harder to control; increasing Kd helps compensate.
    """)

st.caption("Built with Python · NumPy · Matplotlib · Streamlit — UBC CHBE PID Project")
