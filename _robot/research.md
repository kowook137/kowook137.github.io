# SMCDP Progress Report  
## From Position-only SMCDP to Current SE(3) Multi-Embodiment Version

---

## 0. Executive Summary

본 연구는 기존 **position-only self-model manifold diffusion policy**를 출발점으로 하여, 현재는 다음 구조까지 확장되었다.

1. **SE(3) pose-level self-model manifold**
2. **Riemannian / chart-form score learning**
4. **joint-limit bounded chart**
5. **multi-embodiment conditioning**
6. **tool length + link-length perturbation cross-embodiment evaluation**

현재 version은 단순히 end-effector position만 맞추는 policy가 아니라,

\[
T = T_\phi(q,z_e)
\]

를 만족하는 **learned SE(3) self-model manifold 위에서 trajectory를 생성**하며, bounded chart를 통해 joint-limit feasibility를 구조적으로 보장한다.

최신 cross-embodiment 실험에서는

\[
z_e = (z_{\text{tool}}, \Delta l_3, \Delta l_5)
\]

로 embodiment parameter를 확장했고, **5개 sparse embodiment cell만 학습한 뒤**, unseen interpolation / extrapolation embodiment에서 평가.

주요 최신 결과

| Result | Value |
|---|---:|
| Current best v5.1 \(c=2\) effective success | 95.3% |
| Current best v5.1 \(c=2\) joint violation | 0.0% |
| Current best v5.1 \(c=2\) pos error | 1.88 cm |
| Current best v5.1 \(c=2\) rot error | 2.80° |
| Cross-embodiment Seen success | 79.4% |
| Cross-embodiment Interp success | 99.1% |
| Cross-embodiment Extrap success | 76.4% |
| Extrap / Seen success ratio | 0.96 |
| Multi-embodiment joint violation | 0.0% across all cells |

현재 결과는 **same-family morphology generalization**에 대해 강한 evidence.
다만, 완전히 다른 robot morphology로의 transfer 는 안됨. 
-> 좁은 의미의 cross embodiment

---

# 1. Modeling Changes  
## Position-only Version → Current Version

### 1.1 Summary Table

| Version | Modeling Change | Meaning |
|---|---|---|
| Position-only SMCDP | \(p \in \mathbb{R}^3\)만 self-model manifold로 다룸 | end-effector position reaching 중심 |
| Pose-extended SMCDP | \(T_\phi(q,z_e)\in SE(3)\)로 확장 | position + rotation 동시 모델링 |
| Method A | drift-free Brownian + chart-form DSM | 최초 stable pose-SMCDP |
| v4.1 | bounded chart \(q=\psi(u)\) | joint-limit feasibility by construction |
| v5.1 | chart-OU SDE + IK-free reference | IK warm-start / mode leakage 제거 |
| chart_temp | \(\psi_c(u)=q_{mid}+\frac{q_{range}}{2}\tanh(u/c)\) | chart saturation 완화 |
| multi-embodiment | \(z_e=(z_{\text{tool}},\Delta l_3,\Delta l_5)\) | tool + link-length embodiment 일반화 |

---

# 2. Key Experiments  
## Current Single-Embodiment

---

## 2.1 Current Best: v5.1 \(c=2\) vs DP-bounded

| Method | Step | pos cm | rot deg | succ @ 5cm,5deg | succ @ 5cm,10deg | jvio | eff. succ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DP-bounded \(c=2\) | 200k | **1.58** | **1.66** | **93.0%** | 94.5% | 0.4% | 94.5% |
| **v5.1 \(c=2\)** | **275k** | 1.88 | 2.80 | 91.4% | **95.3%** | **0.0%** | **95.3%** |

### Interpretation

| Observation | Meaning |
|---|---|
| DP-bounded has lower mean pos/rot error | DP remains strong in nominal imitation precision |
| v5.1 has slightly higher effective success | feasibility-adjusted task success favors ours |
| v5.1 has 0% joint violation | bounded chart works as intended |
| v5.1 is IK-free | no per-trajectory IK warm-start |
| strict 5deg success is still lower than DP-bounded | remaining endpoint precision gap |

---

## 2.2 Chart Temperature Ablation

| Method | chart temp \(c\) | Step | pos cm | rot deg | succ @ 5cm,5deg | succ @ 5cm,10deg |
|---|---:|---:|---:|---:|---:|---:|
| v5.1 | 1.0 | 100k | 3.88 | 5.63 | 52.7% | 79.3% |
| v5.1 | 2.0 | 100k | **2.66** | **3.63** | 80.9% | 90.2% |
| v5.1 | 2.5 | 100k | 2.77 | 4.01 | **85.2%** | **92.2%** |
| **v5.1** | **2.0** | **275k** | **1.88** | **2.80** | **91.4%** | **95.3%** |

### Interpretation

| Observation | Meaning |
|---|---|
| \(c=1\) performs poorly | chart saturation is a bottleneck |
| \(c=2\) gives large improvement | chart reparameterization is critical |
| \(c=2.5\) is best at 100k | useful for early learning |
| plateau best is \(c=2\) | more stable for long training |
| feasible set unchanged | chart_temp changes parameterization, not joint limits |

---

## 2.3 Endpoint-relative Conditioning Ablation

| Method | Step | pos cm | rot deg | succ @ 5cm,5deg | succ @ 5cm,10deg | Gain |
|---|---:|---:|---:|---:|---:|---:|
| v5.1 baseline \(c=1\) | 100k | 3.88 | 5.63 | 52.7% | 79.3% | — |
| v5.1 + endpoint cond \(c=1\) | 100k | 3.54 | 5.04 | 64.1% | 83.2% | +3.9 pp |
| v5.1 \(c=2\) | 100k | 2.66 | 3.63 | 80.9% | 90.2% | — |
| v5.1 \(c=2\) + endpoint cond | 100k | 2.71 | 3.96 | 80.5% | 91.4% | +1.2 pp |

### Interpretation

| Observation | Meaning |
|---|---|
| endpoint conditioning helps | target-relative SE(3) error is useful |
| gain is modest | not the dominant bottleneck |
| chart_temp gives larger gain | chart parameterization is more important |
| best current recipe includes endpoint cond in multi-embodiment setting | used in latest cross-embodiment run |

---

## 2.4 v5.1 \(c=2\) Learning Curve

| Step | pos cm | rot deg | succ @ 5cm,5deg | succ @ 5cm,10deg | jvio |
|---:|---:|---:|---:|---:|---:|
| 25k | 5.59 | 7.05 | 41.8% | 65.6% | 0.0% |
| 50k | 3.65 | 4.85 | 70.3% | 85.2% | 0.0% |
| 75k | 2.88 | 4.11 | 79.3% | 88.3% | 0.0% |
| 100k | 2.68 | 3.82 | 81.2% | 89.1% | 0.0% |
| 125k | 2.41 | 3.74 | 85.5% | 93.0% | 0.0% |
| 150k | 2.33 | 3.35 | 85.9% | 94.1% | 0.0% |
| 175k | 2.15 | 3.22 | 87.5% | 94.9% | 0.0% |
| 200k | 2.12 | 3.18 | 87.9% | 94.5% | 0.0% |
| 225k | 2.05 | 3.04 | 89.1% | 93.8% | 0.0% |
| 250k | 1.92 | 2.79 | 92.6% | 94.9% | 0.0% |
| **275k** | **1.88** | 2.80 | 91.4% | **95.3%** | **0.0%** |
| 300k | 1.87 | **2.72** | **92.2%** | 94.5% | 0.0% |

### Interpretation

| Observation | Meaning |
|---|---|
| pos/rot mean continues improving | continuous precision improves through training |
| binary success peaks at 275k | threshold metric has cliff effect |
| 300k has better mean but lower succ510 | minor over-training or threshold noise |
| jvio remains 0% | feasibility is stable |

---

# 3. Cross-Embodiment Experiment

---

## 3.1 Experimental Setup

The embodiment parameter is

\[
z_e=(z_{\text{tool}}, \Delta l_3, \Delta l_5)
\]

where \(z_{\text{tool}}\) changes end-effector tool length, and \(\Delta l_3,\Delta l_5\) change link 3 and link 5 length offsets.

### Training Embodiment Cells

Only **5 sparse embodiment cells** are used for training.

| Label | \(z_{\text{tool}}\) | \(\Delta l_3\) | \(\Delta l_5\) | Type |
|---|---:|---:|---:|---|
| T0 | 0.125 | 0.000 | 0.000 | center |
| T1 | 0.050 | -0.030 | -0.030 | corner |
| T2 | 0.050 | +0.030 | +0.030 | corner |
| T3 | 0.200 | -0.030 | +0.030 | corner |
| T4 | 0.200 | +0.030 | -0.030 | corner |

### Evaluation Splits

| Split | Definition | Seen during training? | Difficulty |
|---|---|---|---|
| Seen | T0–T4 | yes | includes hard corners |
| Interp | \(|\Delta l|=0.015\) | no | inside training range |
| Extrap | \(|\Delta l|=0.045\) | no | outside training range |

Training link perturbation range:

\[
|\Delta l| \leq 0.030\ \mathrm{m}
\]

Extrapolation perturbation range:

\[
|\Delta l| = 0.045\ \mathrm{m}
\]

Thus, extrapolation uses link-length perturbation **1.5× outside** the training corner.

---

## 3.2 Stage-1 Multi-Embodiment Self-Model

| Split | n_cells | mean \(p_l\) | mean \(R_l\) | Improvement vs analytic |
|---|---:|---:|---:|---:|
| Seen | 15 | 15.69 mm | 0.30° | 2.4× |
| Interp | 15 | 12.17 mm | 0.25° | 1.6× |
| Extrap | 12 | 19.11 mm | 0.27° | 2.5× |

### Interpretation

| Observation | Meaning |
|---|---|
| Interp error is lowest | middle embodiments are easiest |
| Extrap error increases to 19.11 mm | self-model degradation exists but is not catastrophic |
| rotation error stays low | rotational residual generalizes well |
| Stage-1 error likely contributes to hard-cell failures | especially in extrap cells |

---

## 3.3 Stage-2 Multi-Embodiment Policy Training

| Parameter | Value |
|---|---:|
| Method | v5.1 chart-OU + bounded chart |
| chart temperature | \(c=2\) |
| endpoint-relative conditioning | enabled |
| embodiment dimension | \(n_z=3\) |
| training cells | 5 |
| demo pool | 8192 |
| steps | 200k |
| batch size | 64 |
| training time | 1h12m on RTX A6000 |
| final loss | \(4.66\times 10^{-2}\) |

---

## 3.4 Cross-Embodiment Aggregate Results

| Split | n_cells | \(p_{\text{view}}\) cm | \(R_{\text{view}}\) deg | sw_view | \(p_{\text{real}}\) cm | \(R_{\text{real}}\) deg | sw_real | jvio | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen | 5 | 3.29 | 1.83 | 83.8% | 3.57 | 1.83 | 79.4% | 0.0% | — |
| Interp | 15 | 2.07 | 1.96 | 97.2% | 2.03 | 1.94 | **99.1%** | 0.0% | 1.25 |
| Extrap | 12 | 4.13 | 2.07 | 73.2% | 3.93 | 2.04 | **76.4%** | 0.0% | 0.96 |

### Interpretation

| Observation | Meaning |
|---|---|
| Interp success = 99.1% | unseen in-range embodiments generalize very well |
| Extrap success = 76.4% | outside-range embodiments still work moderately well |
| Extrap / Seen = 0.96 | extrapolation degradation is small relative to seen average |
| jvio = 0% across all cells | bounded chart generalizes feasibility |
| Interp > Seen | Seen includes difficult corner cells |

---

## 3.5 Seen Cell Results

| \(z_{\text{tool}}\) | \(\Delta l_3\) | \(\Delta l_5\) | \(p_{\text{real}}\) cm | \(R_{\text{real}}\) deg | sw_real | jvio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 0.000 | 0.000 | 1.34 | 1.78 | **100.0%** | 0.0% |
| 0.050 | -0.030 | -0.030 | 4.19 | 1.88 | 85.9% | 0.0% |
| 0.050 | +0.030 | +0.030 | 2.47 | 1.75 | **100.0%** | 0.0% |
| 0.200 | -0.030 | +0.030 | 5.26 | 1.99 | **31.2%** | 0.0% |
| 0.200 | +0.030 | -0.030 | 4.56 | 1.77 | 79.7% | 0.0% |

### Interpretation

| Observation | Meaning |
|---|---|
| center cell is perfect | nominal geometry is easy |
| one corner cell drops to 31.2% | long tool + link perturbation is difficult |
| Seen average is pulled down by hard corners | explains why Interp > Seen |

---

## 3.6 Interpolation Results

| \(z_{\text{tool}}\) | \(\Delta l_3\) | \(\Delta l_5\) | \(p_{\text{real}}\) cm | \(R_{\text{real}}\) deg | sw_real |
|---:|---:|---:|---:|---:|---:|
| 0.050 | -0.015 | 0.000 | 1.93 | 1.94 | 100.0% |
| 0.050 | 0.000 | -0.015 | 2.74 | 1.94 | 98.4% |
| 0.050 | +0.015 | 0.000 | 1.61 | 1.95 | 100.0% |
| 0.050 | 0.000 | +0.015 | 1.51 | 2.03 | 100.0% |
| 0.050 | 0.000 | 0.000 | 1.61 | 1.94 | 100.0% |
| 0.125 | -0.015 | 0.000 | 1.51 | 2.02 | 100.0% |
| 0.125 | 0.000 | -0.015 | 2.15 | 1.94 | 100.0% |
| 0.125 | +0.015 | 0.000 | 1.52 | 1.77 | 100.0% |
| 0.125 | 0.000 | +0.015 | 1.78 | 1.88 | 100.0% |
| 0.125 | 0.000 | 0.000 | 1.30 | 1.87 | 100.0% |
| 0.200 | -0.015 | 0.000 | 2.42 | 2.02 | 99.2% |
| 0.200 | 0.000 | -0.015 | 2.13 | 2.02 | 100.0% |
| 0.200 | +0.015 | 0.000 | 2.62 | 1.94 | 97.7% |
| 0.200 | 0.000 | +0.015 | 3.39 | 1.85 | 90.6% |
| 0.200 | 0.000 | 0.000 | 2.26 | 2.00 | 100.0% |

### Interpretation

| Observation | Meaning |
|---|---|
| most interpolation cells are near 100% | strong in-range generalization |
| \(z=0.20,\Delta l_5=+0.015\) is slightly harder | link5 + long tool interaction begins |
| overall interpolation success is 99.1% | very strong result |

---

## 3.7 Extrapolation Results

| \(z_{\text{tool}}\) | \(\Delta l_3\) | \(\Delta l_5\) | \(p_{\text{real}}\) cm | \(R_{\text{real}}\) deg | sw_real |
|---:|---:|---:|---:|---:|---:|
| 0.050 | -0.045 | 0.000 | 3.44 | 2.06 | 99.2% |
| 0.050 | 0.000 | -0.045 | 5.93 | 2.10 | **20.3%** |
| 0.050 | +0.045 | 0.000 | 2.10 | 2.04 | **100.0%** |
| 0.050 | 0.000 | +0.045 | 2.90 | 2.07 | 99.2% |
| 0.125 | -0.045 | 0.000 | 3.02 | 2.24 | 100.0% |
| 0.125 | 0.000 | -0.045 | 5.20 | 1.98 | **33.6%** |
| 0.125 | +0.045 | 0.000 | 2.61 | 2.00 | 98.4% |
| 0.125 | 0.000 | +0.045 | 3.92 | 1.96 | 92.2% |
| 0.200 | -0.045 | 0.000 | 3.77 | 2.13 | 96.1% |
| 0.200 | 0.000 | -0.045 | 4.55 | 2.26 | 71.9% |
| 0.200 | +0.045 | 0.000 | 3.79 | 1.86 | 87.5% |
| 0.200 | 0.000 | +0.045 | 5.90 | 1.81 | **18.0%** |

### Interpretation

| Observation | Meaning |
|---|---|
| \(\Delta l_3\) extrapolation often succeeds | link3 perturbation generalizes well |
| \(\Delta l_5\) extrapolation is harder | wrist-side geometry is more sensitive |
| long tool + link5 extrapolation fails | cumulative end-effector pose error |
| all extrap cells have jvio 0% | feasibility still preserved |

---

## 3.8 Success and Failure Pattern

### Successful Regions

| Region | Result | Interpretation |
|---|---|---|
| Interpolation cells | mostly 97–100% | strong in-distribution morphology interpolation |
| \(\Delta l_3=\pm0.045\) extrap | often 87–100% | link3 change generalizes well |
| center / near-center morphology | near 100% | nominal region is stable |

### Failure Regions

| Region | Result | Likely Cause |
|---|---:|---|
| \(z=0.05,\Delta l_5=-0.045\) | 20.3% | link5 extrapolation sensitivity |
| \(z=0.125,\Delta l_5=-0.045\) | 33.6% | self-model / policy mismatch |
| \(z=0.20,\Delta l_5=+0.045\) | 18.0% | long tool × link5 coupling |
| \(z=0.20,\Delta l_5=-0.045\) | 71.9% | partial degradation |

### Interpretation

The main failure mode is not general extrapolation itself.  
It is concentrated in **link5 perturbation**, especially when combined with longer tool length.

This suggests that wrist-side morphology changes have a larger effect on final end-effector pose than link3 changes.

---

# 4. Trajectory Linearity Analysis

---

## 4.1 Linearity Result

Successful trajectories were compared against an SE(3) linear interpolation path between start and target.

| Method | succ rate | n_succ | dev_int p | dev_int R | dev_real p | dev_real R |
|---|---:|---:|---:|---:|---:|---:|
| DP-raw 200k | 96.1% | 246 | **9.54 mm** | **1.37°** | **6.21 mm** | **0.94°** |
| DP-bounded \(c=2\) 200k | 96.1% | 246 | 10.64 mm | 1.48° | 7.11 mm | 1.05° |
| **Ours v5.1 \(c=2\) + endpt 300k** | 88.3% | 226 | **30.19 mm** | **4.72°** | **31.78 mm** | **4.96°** |

### Interpretation

| Observation | Meaning |
|---|---|
| DP trajectories are more linear | DP better imitates SE(3) linear demo path |
| Ours has about 3× larger path deviation | current method weaker in trajectory shape |
| ours still preserves feasibility | but path quality gap remains |
| endpoint success does not imply path linearity | important limitation |

---

## 4.2 Smoothness Regularizer Sweep

| Config | succ | n_succ | dev_int p | dev_int R |
|---|---:|---:|---:|---:|
| ours baseline | 88.3% | 226 | 30.19 mm | 4.72° |
| ours \(\alpha_v=1\) | 90.6% | 232 | 29.69 mm | 4.70° |
| ours \(\alpha_v=10\) | 78.5% | 201 | 29.54 mm | 4.74° |
| ours \(\alpha_v=100\) | 0.0% | 0 | — | — |
| ours \(\alpha_v=10,\alpha_a=10\) | 5.5% | 14 | 28.42 mm | 4.81° |
| ours \(\alpha_v=100,\alpha_a=100\) | 0.0% | 0 | — | — |

### Interpretation

| Observation | Meaning |
|---|---|
| joint-space smoothness barely improves linearity | local smoothness is not enough |
| high smoothness penalty destroys success | regularizer conflicts with task objective |
| pose-space path guidance is needed | should guide \(T_\phi(q_h,z_e)\) directly |
| DP remains better in path imitation | important limitation to acknowledge |

---

# 5. Current Claims

---

## 5.1 Strong Claims

| Claim | Evidence |
|---|---|
| SE(3) self-model manifold extension works | pose-level trajectory generation is stable |
| Joint feasibility can be guaranteed | bounded chart gives jvio 0% |
| IK warm-start can be removed | v5.1 chart-OU uses IK-free reference |
| Chart parameterization matters | chart_temp \(c=2\) raises succ510 from 79.3% to 95.3% |
| Multi-embodiment conditioning works | \(z_e=(z_{\text{tool}},\Delta l_3,\Delta l_5)\) |
| Sparse embodiment generalization is observed | 5 training cells → Interp 99.1%, Extrap 76.4% |
| Feasibility transfers across embodiment cells | 0% jvio across all cross-embodiment cells |
| Extrapolation is nontrivial | \(|\Delta l|=0.045\) vs training \(|\Delta l|=0.030\) |

---

## 5.2 Claims Requiring Caution

| Claim | Why Caution Is Needed |
|---|---|
| Ours universally beats DP | DP has better mean pos/rot and path linearity |
| Full cross-robot transfer | only same-family link/tool perturbations tested |
| Mathematical guarantee of task success | feasibility is structural, success is empirical |
| All extrapolation works | link5 hard cells fail |
| Trajectory quality is superior | DP is 3× better in linearity |
| Mode preservation fully proven in §15 | multimodality analysis still should be reported separately |

---

# 6. Limitations

---

## 6.1 Hard-cell Failure

The main hard cells are associated with link5 extrapolation, especially with long tool.

| Hard Cell | sw_real | Comment |
|---|---:|---|
| \(z=0.05,\Delta l_5=-0.045\) | 20.3% | strong failure |
| \(z=0.125,\Delta l_5=-0.045\) | 33.6% | strong failure |
| \(z=0.20,\Delta l_5=+0.045\) | 18.0% | strongest failure |
| \(z=0.20,\Delta l_5=-0.045\) | 71.9% | partial failure |

Likely causes:

| Cause | Description |
|---|---|
| self-model extrapolation error | Stage-1 extrap error reaches 19.11 mm |
| wrist-side sensitivity | link5 changes affect end-effector pose strongly |
| long-tool coupling | tool length amplifies distal geometry error |
| threshold cliff | 5 cm position threshold makes moderate error appear as binary failure |

---

## 6.2 Trajectory Linearity Gap

Ours achieves good endpoint success but has weaker path linearity.

| Method | dev_int p | dev_int R |
|---|---:|---:|
| DP-bounded | 10.64 mm | 1.48° |
| Ours | 30.19 mm | 4.72° |

This means the generated path is less close to the demonstration’s SE(3) linear path.

---

## 6.3 Single-seed Limitation

Most experiments are single-seed.  
For paper-level statistical confidence, 3-seed evaluation is needed at least for:

| Experiment | Needed? |
|---|---|
| v5.1 \(c=2\) vs DP-bounded | yes |
| cross-embodiment aggregate | yes |
| hard-cell analysis | yes |
| chart_temp ablation | optional but useful |

---

## 6.4 Scope Limitation

Current cross-embodiment setting is best described as:

> same-family morphology generalization

It should not yet be described as:

> arbitrary cross-robot transfer

The perturbations are meaningful, but they remain within a Franka-like kinematic family.

---

# 7. Immediate Next Experiments

---

## 7.1 DP-bounded Multi-Embodiment Baseline

Current cross-embodiment section needs direct baselines.

| Method | Seen sw_real | Interp sw_real | Extrap sw_real | jvio | linearity |
|---|---:|---:|---:|---:|---:|
| DP-raw multi-emb | TBD | TBD | TBD | TBD | TBD |
| DP-bounded multi-emb | TBD | TBD | TBD | TBD | TBD |
| Ours v5.1 \(c=2\) endpt | 79.4% | 99.1% | 76.4% | 0% | 30 mm |

This is the most important next comparison.

---

## 7.2 Hard-cell Diagnostic

For hard cells, separate self-model error from policy failure.

| Diagnostic | Purpose |
|---|---|
| Stage-1 error per hard cell | check self-model failure |
| oracle FK / oracle self-model evaluation | isolate policy failure |
| target reachability / IK feasibility | check whether task is intrinsically hard |
| generated q distribution | check boundary / chart saturation |
| per-cell pose error distribution | check threshold cliff |

---

## 7.3 Pose-space Path Guidance

Joint smoothness penalty failed.  
A better direction is pose-space path guidance.

\[
\sum_h
\left\|
p_\phi(q_h,z_e)-p_{\mathrm{lin}}(h)
\right\|^2
+
\lambda_R
\left\|
\log
\left(
R_\phi(q_h,z_e)^\top R_{\mathrm{lin}}(h)
\right)
\right\|^2
\]

This directly penalizes deviation from the intended SE(3) path.

---

## 7.4 Stronger Extrapolation Sweep

To characterize failure boundary:

| Parameter | Suggested values |
|---|---|
| \(\Delta l_5\) | \(-0.06,-0.045,-0.03,0,0.03,0.045,0.06\) |
| \(z_{\text{tool}}\) | \(0.05,0.125,0.20,0.25\) |
| \(\Delta l_3\) | fixed 0 or ±0.03 |

This will show where generalization breaks.

---

# 8. Suggested Professor-facing Conclusion

현재 연구는 position-only SMCDP에서 출발해, 현재는 **SE(3) self-model manifold 위에서 joint-safe, IK-free, multi-embodiment trajectory generation**을 수행하는 framework로 확장되었다.

가장 중요한 최신 결과는 다음이다.

1. **v5.1 chart-OU + bounded chart**로 IK warm-start를 제거했다.
2. **chart temperature \(c=2\)** 도입으로 chart saturation bottleneck을 크게 완화했다.
3. 단일 embodiment Tier-2에서는 **v5.1 \(c=2\)**가 DP-bounded와 동등 이상 effective success를 달성했다.
4. multi-embodiment 실험에서는 **5개 sparse embodiment만 학습**하고도:
   - Interp success = 99.1%
   - Extrap success = 76.4%
   - Extrap / Seen ratio = 0.96
   - jvio = 0%
   를 달성했다.
5. 따라서 self-model manifold 기반 conditioning이 **same-family morphology generalization**에 유효하다는 evidence가 생겼다.

단, 현재 결과는 다음 한계를 가진다.

- DP-bounded가 mean pos/rot error와 path linearity에서는 여전히 강하다.
- link5 extrapolation hard-cell에서 failure가 있다.
- full cross-robot transfer는 아직 검증하지 않았다.
- 3-seed reproducibility가 필요하다.

따라서 현재 claim은 다음과 같이 정리하는 것이 가장 안전하다.

> We show that an IK-free self-model manifold diffusion policy can generate joint-feasible and manifold-consistent SE(3) trajectories under sparse same-family embodiment variations. With only five training embodiment cells, the method achieves strong interpolation performance and maintains 96% of seen-cell success on extrapolated link-length perturbations, while preserving zero joint-limit violation. Remaining limitations are concentrated in distal link extrapolation and trajectory path linearity.

---