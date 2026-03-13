# Bioprocess & Control Engineering Simulations

**CHBE 221 · University of British Columbia · Team 22**

Two interactive engineering simulators built in Python and Streamlit, developed as part of a chemical engineering bioprocess design project.

---

## Projects

### 1. Creatine Fed-Batch Bioprocess Simulator
> Simulates creatine production using engineered *Corynebacterium glutamicum* expressing the **Mcgamt** gene (Ptac/IPTG-inducible) in a fed-batch stirred-tank bioreactor.

**Key features:**
- Two-phase fed-batch: growth phase → IPTG induction → creatine production
- Monod kinetics for *C. glutamicum* growth on glucose (μmax = 0.42 h⁻¹)
- Michaelis-Menten kinetics with substrate inhibition for GAMT enzyme activity
- First-order GAMT enzyme induction model (Ptac promoter, LacI repressor)
- Three glucose feeding strategies: constant, exponential, on-demand feedback
- **Scale-up analysis**: Lab (5 L) → Pilot (500 L) → Industrial (50,000 L)
  - Mixing time, oxygen demand, cooling flux, creatine output per batch
- Performance metrics: final creatine titre, volumetric productivity, yield on GAA
- Literature benchmark: 5.42 g/L creatine in 24 h (Li et al., 2024)

**Biological background:**
Creatine is produced via a two-step enzymatic pathway. In this process, *C. glutamicum* is engineered to express GAMT (guanidinoacetate N-methyltransferase) sourced from *Mus caroli* (Mcgamt, 723 bp), codon-optimized for *C. glutamicum*. Guanidinoacetate (GAA) is supplied externally as the direct substrate. GAMT expression is controlled by the Ptac promoter, induced by IPTG addition once target cell density is reached.

**Process conditions (from Part 2 design report):**

| Parameter | Value |
|-----------|-------|
| Temperature | 30 °C |
| pH | 7.0 – 7.2 |
| Dissolved O₂ | ≥ 30% air saturation |
| Reactor type | STR with Rushton turbine |
| Promoter | Ptac (IPTG-inducible) |

---

### 2. PID Flow Rate Controller Simulator
> Simulates a PID feedback control loop regulating flow rate through a pipe/valve system with a first-order process model.

**Key features:**
- First-order process model with configurable time constant, process gain, and dead time
- Live tuning of Kp, Ki, Kd with immediate plot response
- Anti-windup integrator clamping
- Optional inlet pressure disturbance to test rejection
- Performance metrics: rise time, settling time, % overshoot, steady-state error
- 5 plots: flow response, valve position, error, integral term, derivative term

**Engineering background:**

The controller output is:

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d \frac{de}{dt}$$

The process is modelled as a first-order system with dead time:

$$\tau \frac{dF}{dt} = -F + K_p \cdot u + d(t)$$

---

## Getting Started

### Prerequisites
- Python 3.8+
- Anaconda (recommended) or pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/chbe-simulations
cd chbe-simulations
pip install streamlit numpy matplotlib scipy pandas
```

### Running the simulators

**Creatine bioprocess simulator:**
```bash
streamlit run creatine_fedbatch_simulator.py
```

**PID flow controller:**
```bash
streamlit run pid_flow_controller.py
```

Both apps open automatically in your browser at `http://localhost:8501`.

---

## Model Equations

### Creatine Fed-Batch Model

**Cell growth — Monod kinetics:**

$$\mu = \mu_{max} \frac{S}{K_s + S}$$

**Fed-batch mass balances** (D = F_total / V):

$$\frac{dX}{dt} = \mu X - DX$$

$$\frac{dS}{dt} = -\frac{\mu X}{Y_{x/s}} - m_s X - DS + \frac{F_{gluc} \cdot S_{feed}}{V}$$

$$\frac{d[GAA]}{dt} = -v_{GAMT} \cdot X - D \cdot GAA + \frac{F_{GAA} \cdot [GAA]_{feed}}{V}$$

$$\frac{dP}{dt} = v_{GAMT} \cdot X - D \cdot P$$

**GAMT kinetics — Michaelis-Menten with substrate inhibition:**

$$v_{GAMT} = V_{max} \cdot E \cdot \frac{[GAA]}{K_m + [GAA] + [GAA]^2 / K_i}$$

**GAMT enzyme induction:**

$$\frac{dE}{dt} = k_{GAMT}(1 - E) - k_{deg} \cdot E \quad \text{(post-IPTG)}$$

### Scale-Up Correlations

| Parameter | Correlation |
|-----------|-------------|
| Mixing time | t_mix ∝ V^0.33 |
| Cooling area | A_cool ∝ V^0.67 |
| O₂ demand | OUR = μ_max · X · V / Y_O2 |

---

## Key Results

| Metric | Value |
|--------|-------|
| Final creatine titre | ~5.4 g/L (matches Li et al., 2024) |
| Induction time | ~10–12 h (at 15 g DCW/L target) |
| Industrial output (50,000 L) | ~270 kg creatine/batch |

---

## References

1. Li, C. et al. (2024). Efficient biosynthesis of creatine by whole-cell catalysis from guanidinoacetic acid in *Corynebacterium glutamicum*. *Synthetic and Systems Biotechnology*, 9(1), 99–107.
2. Liu, X. et al. (2015). Expression of recombinant protein using *Corynebacterium glutamicum*. *Critical Reviews in Biotechnology*, 36(4), 652–664.
3. Haas, T. et al. (2019). Identifying the Growth Modulon of *Corynebacterium glutamicum*. *Frontiers in Microbiology*, 10.
4. van't Riet, K. (1979). Review of measuring methods and results in nonviscous gas-liquid mass transfer in stirred vessels. *Ind. Eng. Chem. Process Des. Dev.*, 18(3), 357–364.

---

## Project Context

Developed for **CHBE 221** (Biological Aspects of Chemical Engineering) at the **University of British Columbia**, 2026. The bioprocess design covers creatine as a bioproduct, *C. glutamicum* as a production host, reactor selection, medium composition, operating conditions, and expression cassette design.

---

*Built with Python · NumPy · SciPy · Matplotlib · Streamlit*
