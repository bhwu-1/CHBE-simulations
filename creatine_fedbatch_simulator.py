import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
import pandas as pd

st.set_page_config(page_title="Creatine Fed-Batch Simulator", layout="wide")

st.title("Creatine Fed-Batch Bioprocess Simulator")
st.markdown(
    "Simulates creatine production using engineered *C. glutamicum* expressing **Mcgamt** "
    "(Ptac/IPTG-inducible) in a fed-batch stirred-tank bioreactor. "
    "CHBE 221 Team 22 process design."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Biological Parameters")

mu_max    = st.sidebar.slider("Max growth rate umax (h-1)", 0.1, 0.6, 0.42, 0.01,
                               help="Literature: 0.40-0.45 h-1 for C. glutamicum")
Ks        = st.sidebar.slider("Monod constant Ks (g/L)", 0.01, 1.0, 0.15, 0.01,
                               help="Glucose concentration at half-max growth rate")
Yxs       = st.sidebar.slider("Biomass yield Yx/s (g DCW/g glucose)", 0.3, 0.7, 0.50, 0.01)
ms        = st.sidebar.slider("Maintenance coefficient (g glucose/g DCW/h)", 0.0, 0.05, 0.01, 0.001)

st.sidebar.markdown("---")
st.sidebar.header("GAMT / Creatine Kinetics")

k_gamt    = st.sidebar.slider("GAMT induction rate (h-1)", 0.01, 1.0, 0.3, 0.01,
                               help="Rate at which GAMT activity builds up post-IPTG")
Vmax_gamt = st.sidebar.slider("Max GAMT activity Vmax (g creatine/g DCW/h)", 0.1, 2.0, 0.8, 0.05)
Km_gaa    = st.sidebar.slider("Km for GAA (g/L)", 0.01, 1.0, 0.1, 0.01)
Ki_gaa    = st.sidebar.slider("GAA inhibition constant Ki (g/L)", 0.5, 10.0, 3.0, 0.1,
                               help="GAA concentration where substrate inhibition begins")

st.sidebar.markdown("---")
st.sidebar.header("Feed Strategy")

X0        = st.sidebar.slider("Initial biomass X0 (g/L)", 0.1, 2.0, 0.5, 0.1)
S0        = st.sidebar.slider("Initial glucose S0 (g/L)", 5.0, 40.0, 40.0, 1.0)
V0        = st.sidebar.slider("Initial volume V0 (L)", 1.0, 10.0, 5.0, 0.5)
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

    dE     = k_gamt * induced * (1 - E) - 0.02 * E
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

y0  = [X0, S0, 0.0, 0.0, V0, 0.0]
sol = solve_ivp(fed_batch_odes, (0, t_end), y0, args=(params,),
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

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

BG    = "white"
PANEL = "#f8f8f8"
TEXT  = "#222222"
C1    = "#1f4e79"   # dark blue  — biomass / creatine
C2    = "#2e86ab"   # mid blue   — glucose
C3    = "#a23b72"   # plum       — GAA
C4    = "#c73e1d"   # red        — GAMT
C5    = "#3b1f2b"   # dark       — volume
IND   = "#999999"   # grey dashed — induction line

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="semibold", pad=6)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color="#dddddd", linewidth=0.6, linestyle="--")
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

def vline(ax, label=True):
    kw = dict(color=IND, lw=1.2, linestyle="--", alpha=0.9)
    if label:
        ax.axvline(t_induce, label="IPTG induction", **kw)
    else:
        ax.axvline(t_induce, **kw)

# ── Main figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10), facecolor=BG)
gs  = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, X, color=C1, lw=1.8, label="Biomass")
ax1.axhline(X_induce, color=IND, lw=0.9, linestyle=":", alpha=0.7,
            label=f"Induction target ({X_induce} g/L)")
vline(ax1)
ax1.legend(fontsize=7, framealpha=0.5)
style_ax(ax1, "Biomass (C. glutamicum)", "Time (h)", "Biomass (g DCW/L)")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, S, color=C2, lw=1.8)
vline(ax2, label=False)
style_ax(ax2, "Glucose Concentration", "Time (h)", "Glucose (g/L)")

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t, P, color=C1, lw=2.0)
ax3.axhline(5.42, color="#bbbbbb", lw=1, linestyle="--",
            label="Literature target (5.42 g/L)")
vline(ax3)
ax3.legend(fontsize=7, framealpha=0.5)
style_ax(ax3, "Creatine Production", "Time (h)", "Creatine (g/L)")

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t, GA, color=C3, lw=1.8)
vline(ax4, label=False)
style_ax(ax4, "Guanidinoacetate (GAA)", "Time (h)", "GAA (g/L)")

ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(t, E * 100, color=C4, lw=1.8)
vline(ax5, label=False)
style_ax(ax5, "GAMT Enzyme Activity (Ptac/IPTG)", "Time (h)", "GAMT Activity (%)")

ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(t, V, color=C5, lw=1.8)
vline(ax6, label=False)
style_ax(ax6, "Reactor Volume", "Time (h)", "Volume (L)")

fig.patch.set_facecolor(BG)
st.pyplot(fig)
plt.close(fig)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Process Performance")

final_creatine   = P[-1]
final_biomass    = X[-1]
final_volume     = V[-1]
total_creatine_g = final_creatine * final_volume
vol_productivity = total_creatine_g / t_end
gaa_consumed     = F_gaa * GAA_feed * max(0, t_end - t_induce)
creatine_yield   = total_creatine_g / gaa_consumed if gaa_consumed > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Final Creatine Titre",    f"{final_creatine:.2f} g/L",
          delta=f"{final_creatine - 5.42:+.2f} vs lit. (5.42 g/L)")
c2.metric("Final Biomass",           f"{final_biomass:.1f} g DCW/L")
c3.metric("IPTG Induction Time",     f"{t_induce:.1f} h")
c4.metric("Volumetric Productivity", f"{vol_productivity:.2f} g/h")
c5.metric("Creatine Yield on GAA",   f"{creatine_yield:.2f} g/g")

# ── Phase and conditions tables ───────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Process Phases")
    st.markdown(f"""
| Phase | Time | Event |
|-------|------|-------|
| Growth | 0 - {t_induce:.1f} h | Biomass accumulation, GAMT off |
| Induction | {t_induce:.1f} h | IPTG added, Ptac activated |
| Production | {t_induce:.1f} - {t_end} h | GAMT expressed, GAA converted to creatine |
""")
with col2:
    st.subheader("Fixed Operating Conditions")
    st.markdown("""
| Parameter | Value | Reference |
|-----------|-------|-----------|
| Temperature | 30 C | Part 2 s2.4 |
| pH | 7.0 - 7.2 | Part 2 s2.4 |
| Dissolved O2 | >= 30% air sat. | Part 2 s2.4 |
| Reactor | STR, Rushton turbine | Part 2 s2.3 |
| Promoter | Ptac (IPTG-inducible) | Part 1 s2.3 |
| GAMT gene | Mcgamt (M. caroli, 723 bp) | Part 1 s2.3 |
""")

# ── Scale-Up Analysis ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Scale-Up Analysis")
st.markdown(
    "Performance projections scaling from lab bench to industrial production. "
    "Key engineering constraints shift at each order of magnitude."
)

scale_rows = []
for name, V_scale in [("Lab (5 L)", 5), ("Pilot (500 L)", 500), ("Industrial (50,000 L)", 50000)]:
    mix_time    = 2.5 * (V_scale ** 0.33)
    Q_gen       = 80 * V_scale
    A_cool      = 4.84 * (V_scale ** 0.67)
    heat_flux   = Q_gen / (A_cool * 1000)
    OUR         = 60 * final_biomass * V_scale / 1000
    creatine_kg = final_creatine * V_scale / 1000
    challenge   = (
        "Baseline — good O2 transfer, easy mixing" if V_scale <= 5 else
        "Mixing gradients appear, impeller optimization required" if V_scale <= 500 else
        "Heat removal critical, O2 transfer limiting, sterility risk"
    )
    scale_rows.append({
        "Scale":                      name,
        "Volume (L)":                 f"{V_scale:,}",
        "Creatine output (kg/batch)": f"{creatine_kg:.2f}",
        "Mixing time (s)":            f"{mix_time:.0f}",
        "O2 demand (mol/h)":          f"{OUR:.0f}",
        "Cooling flux (kW/m2)":       f"{heat_flux:.2f}",
        "Key constraint":             challenge,
    })

df_scale = pd.DataFrame(scale_rows).set_index("Scale")
st.dataframe(df_scale, use_container_width=True)

fig2, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG)
vols        = [5, 500, 50000]
creatine_out= [final_creatine * v / 1000 for v in vols]
mix_times   = [2.5 * (v ** 0.33) for v in vols]
heat_fluxes = [(80 * v) / (4.84 * (v ** 0.67) * 1000) for v in vols]
xlabels     = ["Lab\n5 L", "Pilot\n500 L", "Industrial\n50,000 L"]
bar_colors  = [C2, C3, C4]

for ax, ydata, ylabel, title in zip(
    axes,
    [creatine_out, mix_times, heat_fluxes],
    ["Creatine (kg/batch)", "Mixing Time (s)", "Cooling Flux (kW/m2)"],
    ["Creatine Output per Batch", "Mixing Time vs Scale", "Cooling Demand vs Scale"]
):
    bars = ax.bar(xlabels, ydata, color=bar_colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, ydata):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                f"{val:.2f}", ha="center", color=TEXT, fontsize=9)
    ax.set_facecolor(PANEL)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="semibold", pad=6)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

fig2.patch.set_facecolor(BG)
fig2.tight_layout(pad=2)
st.pyplot(fig2)
plt.close(fig2)

with st.expander("Scale-Up Engineering Notes"):
    st.markdown("""
**Mixing time** grows with volume (~V^0.33). At 50,000 L, mixing takes roughly 90 seconds
compared to 3 seconds at lab scale. This creates concentration gradients of glucose, GAA, and
dissolved oxygen across the reactor — cells near the feed inlet experience different conditions
than those elsewhere. This is the primary motivation for specifying a Rushton turbine and baffles
in the reactor design (Part 2, s2.3).

**Oxygen transfer** becomes the rate-limiting step at industrial scale. At high cell densities,
the oxygen uptake rate can exceed the reactor supply capacity, causing DO to drop below the 30%
threshold. Cascade control via agitation RPM and airflow is used to manage this (Part 2, s2.4).

**Heat removal** — metabolic heat generation scales linearly with volume, but cooling surface area
only scales with V^0.67. At 50,000 L, a cooling jacket alone is insufficient and internal coils
are required.

**Sterility** risk increases with scale due to more connections, longer run times, and larger
surface areas. Addressed through SIP/CIP, sterile gas filtration, and closed sampling systems
(Part 2, s2.3).
""")

# ── Model equations ───────────────────────────────────────────────────────────
with st.expander("Model Equations and Assumptions"):
    st.markdown(r"""
**Cell growth — Monod kinetics:**
$$\mu = \mu_{max} \frac{S}{K_s + S}$$

**Fed-batch mass balances** (D = F_total / V):
$$\frac{dX}{dt} = \mu X - DX$$
$$\frac{dS}{dt} = -\frac{\mu X}{Y_{x/s}} - m_s X - DS + \frac{F_{gluc} \cdot S_{feed}}{V}$$
$$\frac{d[GAA]}{dt} = -v_{GAMT} \cdot X - D \cdot GAA + \frac{F_{GAA} \cdot [GAA]_{feed}}{V}$$
$$\frac{dP}{dt} = v_{GAMT} \cdot X - D \cdot P$$

**GAMT — Michaelis-Menten with substrate inhibition:**
$$v_{GAMT} = V_{max} \cdot E \cdot \frac{[GAA]}{K_m + [GAA] + [GAA]^2 / K_i}$$

**GAMT induction (first-order, post-IPTG):**
$$\frac{dE}{dt} = k_{GAMT}(1 - E) - k_{deg} \cdot E$$

**Assumptions:** SAM (methyl donor) non-limiting; O2 non-limiting (DO >= 30% via cascade control);
isothermal at 30 C; pH 7.0 maintained by PID; no product inhibition by creatine at these titres.
""")

st.caption("CHBE 221 Team 22 — C. glutamicum Creatine Fed-Batch Simulator")
