# Creatine Fed-Batch Bioprocess Simulator

A simulation of creatine production using engineered *Corynebacterium glutamicum* expressing the **Mcgamt** gene in a fed-batch stirred-tank bioreactor. Built as a supplementary computational tool for CHBE 221 at the University of British Columbia.

---

## Background

Creatine is produced via a two-step enzymatic pathway. In this process, *C. glutamicum* is engineered to express GAMT (guanidinoacetate N-methyltransferase), sourced from *Mus caroli* (Mcgamt, 723 bp) and codon-optimized for bacterial expression. Guanidinoacetate (GAA) is supplied externally as the direct substrate. GAMT expression is controlled by the Ptac promoter and induced by IPTG addition once the target cell density is reached, decoupling the growth and production phases within the same vessel.

The simulation is grounded in the kinetic and process parameters from the CHBE 221 Team 22 design reports, with a literature benchmark of 5.42 g/L creatine over 24 hours (Li et al., 2024).

---

## Features

- Two-phase fed-batch: growth phase with no GAMT expression, followed by IPTG induction and creatine production
- Monod kinetics for *C. glutamicum* growth on glucose (umax = 0.42 h-1)
- Michaelis-Menten kinetics with substrate inhibition for GAMT activity
- First-order GAMT enzyme induction model (Ptac promoter, LacI repressor)
- Three glucose feeding strategies: constant rate, exponential, on-demand feedback
- Scale-up analysis from lab (5 L) to pilot (500 L) to industrial (50,000 L), including mixing time, oxygen demand, and cooling flux projections
- Interactive sliders for all kinetic and process parameters
- Performance metrics: final creatine titre, volumetric productivity, yield on GAA, induction time

---

## Getting Started

**Requirements:** Python 3.8+, Anaconda recommended

```bash
git clone https://github.com/YOUR_USERNAME/creatine-simulator
cd creatine-simulator
pip install streamlit numpy matplotlib scipy pandas
streamlit run creatine_fedbatch_simulator.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Model Equations

**Cell growth — Monod kinetics:**

$$\mu = \mu_{max} \frac{S}{K_s + S}$$

**Fed-batch mass balances** (D = F_total / V):

$$\frac{dX}{dt} = \mu X - DX$$

$$\frac{dS}{dt} = -\frac{\mu X}{Y_{x/s}} - m_s X - DS + \frac{F_{gluc} \cdot S_{feed}}{V}$$

$$\frac{d[GAA]}{dt} = -v_{GAMT} \cdot X - D \cdot GAA + \frac{F_{GAA} \cdot [GAA]_{feed}}{V}$$

$$\frac{dP}{dt} = v_{GAMT} \cdot X - D \cdot P$$

**GAMT kinetics — Michaelis-Menten with substrate inhibition:**

$$v_{GAMT} = V_{max} \cdot E \cdot \frac{[GAA]}{K_m + [GAA] + [GAA]^2 / K_i}$$

**GAMT enzyme induction (first-order, post-IPTG):**

$$\frac{dE}{dt} = k_{GAMT}(1 - E) - k_{deg} \cdot E$$

**Scale-up correlations:**

| Parameter | Relationship |
|-----------|-------------|
| Mixing time | t_mix proportional to V^0.33 |
| Cooling surface area | A proportional to V^0.67 |
| O2 demand | OUR = mu * X * V / Y_O2 |

---

## Key Results

| Metric | Value |
|--------|-------|
| Final creatine titre | ~5.4 g/L (consistent with Li et al., 2024) |
| Induction time | ~10-12 h at 15 g DCW/L target biomass |
| Industrial output (50,000 L) | ~270 kg creatine per batch |

---

## Process Conditions

| Parameter | Value | Source |
|-----------|-------|--------|
| Temperature | 30 C | Part 2, s2.4 |
| pH | 7.0 - 7.2 | Part 2, s2.4 |
| Dissolved O2 | >= 30% air saturation | Part 2, s2.4 |
| Reactor | STR with Rushton turbine | Part 2, s2.3 |
| Promoter | Ptac (IPTG-inducible) | Part 1, s2.3 |
| GAMT source | Mcgamt, M. caroli, 723 bp | Part 1, s2.3 |

---

## References

1. Li, C. et al. (2024). Efficient biosynthesis of creatine by whole-cell catalysis from guanidinoacetic acid in *Corynebacterium glutamicum*. *Synthetic and Systems Biotechnology*, 9(1), 99-107.
2. Liu, X. et al. (2015). Expression of recombinant protein using *Corynebacterium glutamicum*. *Critical Reviews in Biotechnology*, 36(4), 652-664.
3. Haas, T. et al. (2019). Identifying the Growth Modulon of *Corynebacterium glutamicum*. *Frontiers in Microbiology*, 10.
4. van't Riet, K. (1979). Review of measuring methods and results in nonviscous gas-liquid mass transfer in stirred vessels. *Industrial and Engineering Chemistry Process Design and Development*, 18(3), 357-364.

---

## Project Context

Developed for CHBE 221 (Biological Aspects of Chemical Engineering) at the University of British Columbia, 2026. Team 22: Keshan Baduge, Alex Chau, Silas Freeman, Willis Wong, Ben Wu.

Built with Python, NumPy, SciPy, Matplotlib, Streamlit.
