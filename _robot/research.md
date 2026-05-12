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
