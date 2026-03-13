import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Creatine Fed-Batch Simulator", layout="wide")

st.title("🧬 Creatine Fed-Batch Bioprocess Simulator")
st.markdown(
    "Simulates creatine production using engineered *C. glutamicum* expressing **Mcgamt** "
    "(Ptac/IPTG-inducible) in a fed-batch stirred-tank bioreactor. "
    "Based on CHBE 221 Team 22 process design."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🦠 Biological Parameters")

mu_max   = st.sidebar.slider("Max growth rate μmax (h⁻¹)", 0.1, 0.6, 0.42, 0.01,
                              help="Literature: 0.40–0.45 h⁻¹ for C. glutamicum")
Ks       = st.sidebar.slider("Monod constant Ks (g/L)", 0.01, 1.0, 0.15, 0.01,
                              help="Glucose concentration at half-max growth rate")
Yxs      = st.sidebar.slider("Biomass yield Yx/s (g DCW/g glucose)", 0.3, 0.7, 0.50, 0.01,
                              help="Grams of biomass per gram of glucose consumed")
ms       = st.sidebar.slider("Maintenance coefficient (g glucose/g DCW/h)", 0.0, 0.05, 0.01, 0.001)

st.sidebar.markdown("---")
st.sidebar.header("⚗️ GAMT / Creatine Kinetics")

k_gamt    = st.sidebar.slider("GAMT induction rate (h⁻¹)", 0.01, 1.0, 0.3, 0.01,
                               help="Rate at which GAMT activity builds up post-IPTG")
Vmax_gamt = st.sidebar.slider("Max GAMT activity Vmax (g creatine/g DCW/h)", 0.1, 2.0, 0.8, 0.05)
Km_gaa    = st.sidebar.slider("Km for GAA (g/L)", 0.01, 1.0, 0.1, 0.01)
Ki_gaa    = st.sidebar.slider("GAA inhibition constant Ki (g/L)", 0.5, 10.0, 3.0, 0.1,
                               help="GAA concentration where substrate inhibition begins")

st.sidebar.markdown("---")
st.sidebar.header("🏭 Feed Strategy")

X0        = st.sidebar.slider("Initial biomass X₀ (g/L)", 0.1, 2.0, 0.5, 0.1)
S0        = st.sidebar.slider("Initial glucose S₀ (g/L)", 5.0, 40.0, 40.0, 1.0)
V0        = st.sidebar.slider("Initial volume V₀ (L)", 1.0, 10.0, 5.0, 0.5)
X_induce  = st.sidebar.slider("Induction biomass target (g/L)", 5.0, 30.0, 15.0, 1.0,
                               help="IPTG added when biomass reaches this density")

feed_mode = st.sidebar.selectbox("Glucose feed strategy",
                                  ["Constant rate", "Exponential", "On-demand (feedback)"])
F_gluc    = st.sidebar.slider("Glucose feed rate (L/h)", 0.0, 0.5, 0.05, 0.005)
S_feed    = st.sidebar.slider("Feed glucose concentration (g/L)", 100.0, 500.0, 300.0, 10.0)
F_gaa     = st.sidebar.slider("GAA feed rate after induction (L/h)", 0.0, 0.3, 0.05, 0.005)
GAA_feed  = st.sidebar.slider("Feed GAA concentration (g/L)", 10.0, 100.0, 50.0, 5.0)
t_end     = st.sidebar.slider("Simulation duration (h)", 12, 48, 24, 1)

# ── ODE System ────────────────────────────────────────────────────────────────
# State: [X, S, GAA, P (creatine), V, E (GAMT activity 0-1)]

def fed_batch_odes(t, y, params):
    X, S, GAA, P, V, E = y
    X = max(X, 0); S = max(S, 0); GAA = max(GAA, 0); P = max(P, 0); V = max(V, 1e-6)

    (mu_max, Ks, Yxs, ms, k_gamt,
     Vmax_gamt, Km_gaa, Ki_gaa,
     F_gluc, S_feed, F_gaa, GAA_feed,
     X_induce, feed_mode) = params

    mu = mu_max * S / (Ks + S)
    induced = 1.0 if X >= X_induce else 0.0

    if feed_mode == "Exponential":
        F_g = F_gluc * np.exp(0.08 * t)
    elif feed_mode == "On-demand (feedback)":
        F_g = F_gluc if S < 5.0 else 0.0
    else:
        F_g = F_gluc

    F_g_act   = F_g * induced + F_gluc * 0.3 * (1 - induced)
    F_gaa_act = F_gaa * induced
    F_total   = F_g_act + F_gaa_act
    D         = F_total / V

    dE   = k_gamt * induced * (1 - E) - 0.02 * E
    v_gamt = Vmax_gamt * E * GAA / (Km_gaa + GAA + (GAA**2) / Ki_gaa)

    dX   =  mu * X - D * X
    dS   = -mu * X / Yxs - ms * X - D * S + F_g_act * S_feed / V
    dGAA = -v_gamt * X - D * GAA + F_gaa_act * GAA_feed / V
    dP   =  v_gamt * X - D * P
    dV   =  F_total

    return [dX, dS, dGAA, dP, dV, dE]

# ── Run ───────────────────────────────────────────────────────────────────────
params = (mu_max, Ks, Yxs, ms, k_gamt, Vmax_gamt, Km_gaa, Ki_gaa,
          F_gluc, S_feed, F_gaa, GAA_feed, X_induce, feed_mode)

y0     = [X0, S0, 0.0, 0.0, V0, 0.0]
sol    = solve_ivp(fed_batch_odes, (0, t_end), y0, args=(params,),
                   t_eval=np.linspace(0, t_end, 600), method="RK45", max_step=0.05)

t  = sol.t
X  = np.maximum(sol.y[0], 0)
S  = np.maximum(sol.y[1], 0)
GA = np.maximum(sol.y[2], 0)
P  = np.maximum(sol.y[3], 0)
V  = sol.y[4]
E  = np.clip(sol.y[5], 0, 1)

ind_idx  = np.where(X >= X_induce)[0]
t_induce = t[ind_idx[0]] if len(ind_idx) > 0 else t_end

# ── Colours ───────────────────────────────────────────────────────────────────
BG = "#0e1117"; PANEL = "#1a1d27"; TEXT = "#e0e0e0"
CYAN = "#00c8ff"; GREEN = "#00e676"; ORANGE = "#ff9800"
RED = "#ff4444"; PURPLE = "#bb86fc"; YELLOW = "#ffd54f"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
    ax.tick_params(colors=TEXT, labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#333")
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color="#2a2d3a", linewidth=0.5, linestyle="--")

def vline(ax):
    ax.axvline(t_induce, color=YELLOW, lw=1.2, linestyle="--", alpha=0.75, label="IPTG induction")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10), facecolor=BG)
gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, X, color=GREEN, lw=2, label="Biomass")
ax1.axhline(X_induce, color=YELLOW, lw=1, linestyle=":", alpha=0.8, label=f"Induction target ({X_induce} g/L)")
vline(ax1)
ax1.set_xlabel("Time (h)"); ax1.set_ylabel("Biomass (g DCW/L)")
ax1.legend(facecolor=PANEL, edgecolor="#444", labelcolor=TEXT, fontsize=7)
style_ax(ax1, "Biomass Growth (C. glutamicum)")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, S, color=ORANGE, lw=2)
vline(ax2)
ax2.set_xlabel("Time (h)"); ax2.set_ylabel("Glucose (g/L)")
style_ax(ax2, "Glucose Concentration")

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t, P, color=CYAN, lw=2.5)
ax3.axhline(5.42, color=RED, lw=1, linestyle="--", alpha=0.8, label="Literature target (5.42 g/L)")
vline(ax3)
ax3.set_xlabel("Time (h)"); ax3.set_ylabel("Creatine (g/L)")
ax3.legend(facecolor=PANEL, edgecolor="#444", labelcolor=TEXT, fontsize=7)
style_ax(ax3, "Creatine Production")

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t, GA, color=PURPLE, lw=2)
vline(ax4)
ax4.set_xlabel("Time (h)"); ax4.set_ylabel("GAA (g/L)")
style_ax(ax4, "Guanidinoacetate (GAA) — Substrate for GAMT")

ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(t, E * 100, color=RED, lw=2)
vline(ax5)
ax5.set_xlabel("Time (h)"); ax5.set_ylabel("GAMT Activity (%)")
style_ax(ax5, "GAMT Enzyme Expression (Ptac/IPTG-induced)")

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(t, V, color=YELLOW, lw=2)
vline(ax6)
ax6.set_xlabel("Time (h)"); ax6.set_ylabel("Volume (L)")
style_ax(ax6, "Reactor Volume (Fed-Batch)")

fig.patch.set_facecolor(BG)
st.pyplot(fig)
plt.close(fig)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Process Performance Metrics")

final_creatine   = P[-1]
final_biomass    = X[-1]
final_volume     = V[-1]
total_creatine_g = final_creatine * final_volume
vol_productivity = total_creatine_g / t_end
gaa_consumed     = F_gaa * GAA_feed * max(0, t_end - t_induce)
creatine_yield   = total_creatine_g / gaa_consumed if gaa_consumed > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Final Creatine Titre", f"{final_creatine:.2f} g/L",
          delta=f"{final_creatine - 5.42:+.2f} vs lit. (5.42 g/L)")
c2.metric("Final Biomass",        f"{final_biomass:.1f} g DCW/L")
c3.metric("IPTG Induction Time",  f"{t_induce:.1f} h")
c4.metric("Volumetric Productivity", f"{vol_productivity:.2f} g/h")
c5.metric("Creatine Yield on GAA",   f"{creatine_yield:.2f} g/g")

# ── Phase table ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔬 Process Phases")
    st.markdown(f"""
| Phase | Time | Event |
|-------|------|-------|
| **Growth** | 0 → {t_induce:.1f} h | Biomass accumulation, GAMT OFF |
| **Induction** | {t_induce:.1f} h | IPTG added → Ptac activated |
| **Production** | {t_induce:.1f} → {t_end} h | GAMT expressed, GAA → Creatine |
""")
with col2:
    st.subheader("⚙️ Fixed Operating Conditions")
    st.markdown("""
| Parameter | Value | Reference |
|-----------|-------|-----------|
| Temperature | 30 °C | Part 2 §2.4 |
| pH | 7.0 – 7.2 | Part 2 §2.4 |
| DO | ≥ 30% air sat. | Part 2 §2.4 |
| Reactor | STR, Rushton turbine | Part 2 §2.3 |
| Promoter | Ptac (IPTG-inducible) | Part 1 §2.3 |
| GAMT gene | Mcgamt (M. caroli, 723 bp) | Part 1 §2.3 |
""")

# ── Scale-up Analysis ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Scale-Up Analysis")
st.markdown(
    "How does this process perform as we scale from a **5 L lab bench** → "
    "**500 L pilot plant** → **50,000 L industrial** bioreactor? "
    "Key engineering challenges change dramatically at each scale."
)

scales = {
    "Lab (5 L)":        {"V": 5,      "scale_factor": 1.0},
    "Pilot (500 L)":    {"V": 500,    "scale_factor": 100.0},
    "Industrial (50,000 L)": {"V": 50000, "scale_factor": 10000.0},
}

# Engineering correlations for scale-up
# Power per unit volume (P/V) kept constant is common industrial approach
# kLa scales with (P/V)^0.4 * vs^0.5  (van't Riet correlation)
# Mixing time scales with V^0.33
# Heat removal area/volume ratio drops with scale

P_V_lab   = 3000   # W/m³ — typical lab bioreactor
vs_lab    = 0.05   # m/s — superficial gas velocity

scale_rows = []
for name, props in scales.items():
    V_scale = props["V"]
    sf      = props["scale_factor"]

    # Keep P/V constant → same kLa (conservative assumption)
    kLa      = 0.3 * (P_V_lab ** 0.4) * (vs_lab ** 0.5) / 10  # h⁻¹ normalized
    mix_time = 2.5 * (V_scale ** 0.33)                          # seconds, empirical

    # Heat generation scales with volume; area scales with V^0.67
    Q_gen    = 80 * V_scale          # W (metabolic heat ~80 W/L at high density)
    A_cool   = 4.84 * (V_scale ** 0.67)  # m² cooling area
    heat_flux= Q_gen / (A_cool * 1000)   # kW/m²

    # Oxygen demand
    OUR      = 60 * final_biomass * V_scale / 1000  # mol O2/h

    # Creatine output
    creatine_kg = final_creatine * V_scale / 1000

    scale_rows.append({
        "Scale":               name,
        "Volume (L)":          f"{V_scale:,}",
        "Creatine output (kg/batch)": f"{creatine_kg:.2f}",
        "Mixing time (s)":     f"{mix_time:.0f}",
        "O₂ demand (mol/h)":   f"{OUR:.0f}",
        "Cooling flux (kW/m²)":f"{heat_flux:.2f}",
        "Key challenge":       (
            "Baseline — good O₂ transfer, easy mixing"
            if V_scale <= 5 else
            "Mixing gradients appear, need impeller optimization"
            if V_scale <= 500 else
            "Heat removal critical, O₂ transfer limiting, sterility risk"
        )
    })

import pandas as pd
df_scale = pd.DataFrame(scale_rows).set_index("Scale")
st.dataframe(df_scale, use_container_width=True)

# Scale-up chart
fig2, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG)
vols        = [5, 500, 50000]
creatine_out= [final_creatine * v / 1000 for v in vols]
mix_times   = [2.5 * (v ** 0.33) for v in vols]
heat_fluxes = [(80 * v) / (4.84 * (v ** 0.67) * 1000) for v in vols]
labels      = ["Lab\n5 L", "Pilot\n500 L", "Industrial\n50,000 L"]
colors      = [CYAN, ORANGE, RED]

for ax, ydata, ylabel, title in zip(
    axes,
    [creatine_out, mix_times, heat_fluxes],
    ["Creatine (kg/batch)", "Mixing Time (s)", "Cooling Flux (kW/m²)"],
    ["Creatine Output per Batch", "Mixing Time vs Scale", "Cooling Demand vs Scale"]
):
    bars = ax.bar(labels, ydata, color=colors, edgecolor="#333", width=0.5)
    for bar, val in zip(bars, ydata):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                f"{val:.2f}", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    ax.set_facecolor(PANEL); ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=7)
    ax.grid(True, axis="y", color="#2a2d3a", linewidth=0.5, linestyle="--")

fig2.patch.set_facecolor(BG)
fig2.tight_layout(pad=2)
st.pyplot(fig2)
plt.close(fig2)

with st.expander("💡 Scale-Up Engineering Notes", expanded=False):
    st.markdown("""
    ### Why Scale-Up Is Non-Trivial

    **Mixing time** grows with volume (~V⁰·³³). At 50,000 L, mixing takes ~90 seconds vs ~3 seconds
    at lab scale. This creates **concentration gradients** of glucose, GAA, and dissolved oxygen —
    cells near the feed inlet see different conditions than those far away. This is why your report
    specifies a Rushton turbine (23–77% kLa enhancement) and baffles.

    **Oxygen transfer** becomes the limiting factor at industrial scale. The oxygen uptake rate (OUR)
    of a dense *C. glutamicum* culture can exceed the reactor's capacity to supply O₂, causing
    DO to drop below the 30% threshold specified in your design. Solutions include:
    - Increase agitation RPM (cascade DO control, Part 2 §2.4)
    - Increase airflow rate
    - Enrich with pure O₂ if needed

    **Heat removal** — metabolic heat generation scales with volume (V), but cooling area only
    scales with V⁰·⁶⁷. At 50,000 L, a cooling jacket alone is insufficient — internal cooling
    coils are required, consistent with your reactor design specification.

    **Sterility risk** increases with scale. More inlet/outlet connections, longer run times,
    and larger surface areas all increase contamination probability — addressed in your SIP/CIP
    and sterile sampling strategy (Part 2 §2.3).
    """)

# ── Equations ─────────────────────────────────────────────────────────────────
with st.expander("📚 Model Equations & Assumptions", expanded=False):
    st.markdown(r"""
### Kinetic Model

**Cell growth — Monod kinetics:**
$$\mu = \mu_{max} \frac{S}{K_s + S}$$

**Fed-batch ODEs (D = F_total / V):**
$$\frac{dX}{dt} = \mu X - DX$$
$$\frac{dS}{dt} = -\frac{\mu X}{Y_{x/s}} - m_s X - DS + \frac{F_{gluc} \cdot S_{feed}}{V}$$
$$\frac{d[GAA]}{dt} = -v_{GAMT} \cdot X - D \cdot GAA + \frac{F_{GAA} \cdot GAA_{feed}}{V}$$
$$\frac{dP}{dt} = v_{GAMT} \cdot X - D \cdot P$$

**GAMT — Michaelis-Menten with substrate inhibition:**
$$v_{GAMT} = V_{max} \cdot E \cdot \frac{[GAA]}{K_m + [GAA] + [GAA]^2 / K_i}$$

**GAMT enzyme induction (first-order, post-IPTG):**
$$\frac{dE}{dt} = k_{GAMT}(1 - E) - k_{deg} \cdot E$$

### Key Assumptions
- SAM (methyl donor) is non-limiting — native C. glutamicum metabolism sufficient (Part 1 §2.1)
- Oxygen is non-limiting (DO ≥ 30% via cascade control, Part 2 §2.4)
- Isothermal at 30°C, pH 7.0 maintained by PID control (Part 2 §2.4)
- Codon-optimized Mcgamt ensures efficient GAMT translation (Part 1 §2.3)
- No significant product inhibition by creatine at these titres
""")

st.caption("CHBE 221 Team 22 · C. glutamicum Creatine Fed-Batch Simulator · Python / SciPy / Streamlit")
