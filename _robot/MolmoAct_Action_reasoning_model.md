---
title: "Molmo ACT"
date: 2026-01-09
tags: [VLA, Robotics]
toc: true
mathjax: true
---
> https://github.com/allenai/MolmoAct

## 1. 서론  

### 1.1 problem formulation  

VLA 의 일반적 구조: VLM(Vision Encoder + LLM backbone) 에서 LLM Backbone 의 decoder 부분에서 일부를 action token 으로 substitute 하여 Action sequence Generation.(RT-1, RT-2, Open-VLA 등)  

이때 VLM은 2D RGB 이미지로 주로 학습되기 때문에 spatial reasoning 에 매우 취약하다. CoT-VLA 류의 language base spatial reasoning 역시 공간 정보를 충분히 표현하지 못하기 때문에 본질적인 해결책이 될 수 없음.  

VLA 의 한계점 3가지  
1) 환경 변화에 취약함  
2) Embodiment 변화에 취약함  
3) Opaque. 왜 해당 Action을 도출했는지 Explainable 하지 않음.

## 2. 방법론  

MolmoAct-7B-D

### 2.1 MolmoAct architecture  

Molmo VLM architecture 를 그대로 활용함.

{% include figure
   image_path="/assets/images/robot/molmo_vlm.png"
   alt="Molmo VLM"
   caption="Molmo VLM architecture"
   class="align-center"
%}   

Molmo VLM architecture는 보통의 VLM 과 같이, Vision Encoder 와 LLM 의 결합으로 구성됨.  

### 2.2 MolmoAct 구성  

- Vision Encoder: SigLIP2  
- LLM backbone: Qwen 2.5 7B
- Mapping: Language instruction, image to Joint command

### 2.3 학습 데이터  

OXE 일부(정제된 데이터), BC-Z, V2, RT-1, Molmo ACT dataset (curated)  

### 2.4 GPU hours  

9,216 GPU hours, GR00T 대비 5배 절감, 256 H

### 2.5 구현상의 특징  
 
#### 2.5.1 BPE level 을 고려한 action token substitution  

기존의 VLA는 vocal tail 중에서 사용되지 않는 256개를 임의로 선정하여 action bin 에 할당하는 방식을 사용함.  

기존 방식은 action 은 물리적으로 서로 유사한/유사하지 않은 action 이라는 관계성이 존재하는데, action bin 에 vocal tail 중 아무런 256개를 할당하면 embedding space 상에서 유사한 action 이 가까이 위치하지 않음. 이러한 상태는 학습에 매우 불리함.  

MolmoAct 는 action bin 으로 할당할 vocal tail 을 선정할 때, 다음의 3가지 조건을 만족하는 token 을 선정함.  

1) 단일 byte token
(token ID 에 해당하는 string 이 1 byte 로 표현되는 token)  

2) byte level 에서 연속된 token 집합  
(예컨대 byte 에서 1000100, 1000101, 1000110 ... 이런 토큰들)

3) 유사한 action 과 vocal tail token 의 monotonic 한 mapping  

{% include figure
   image_path="/assets/images/robot/action_token_subst.png"
   alt="action token subst"
   caption="MolmoAct 의 Action Token 할당 방식"
   class="align-center"
%}  

이렇게 action token을 만들면, 기존의 VLA와 다르게 물리적으로 비슷한 action 은 tokenizer 의 embedding space 에서도 가까워져 학습 초기 gradient 가 안정적으로 형성됨.  

따라서 안정적인 학습, 빠른 수렴, 더 부드러운 동작이 가능함.
(왜냐하면 이를 통해 softmax distribution 이 의미를 가지게 된다.)  

#### 2.5.2 Spatial Reasoning 을 위한 Depth perception token  

(1) Teacher (Depth Tokenization)

$$
\text{Model} \equiv \text{teacher model}
$$

$$
D \in \mathbb{R}^{H \times W}
\quad (\text{dense depth map})
$$

$$
\mathrm{VQ\text{-}VAE} : D \mapsto z^{\text{depth}}
$$

$$
z^{\text{depth}} = (z_1, \dots, z_L),
\qquad z_i \in \{1, \dots, 128\}
$$

$$
\text{Deterministic encoding: }
z^{\text{depth}} \mapsto d
$$

$$
d =
\langle \text{DEPTH\_START} \rangle,
\langle \text{DEPTH\_}z_1 \rangle,
\langle \text{DEPTH\_}z_2 \rangle,
\dots,
\langle \text{DEPTH\_END} \rangle
$$

---

  (2) MolmoAct (Distillation Target)

$$
(I, a) \xmapsto{\text{MolmoAct}} \hat{d}^{\text{depth}}
$$

- $\mathcal{C} = \{c_1, \ldots, c_N\},\; c_k \in \mathbb{R}^d$ 인 코드북에 대해서,
  각 $c_n$ 은 RGB 상에서 depth map 의 국소 패턴을 나타내는 vector.
  (총 128개 패턴)

- $M = 100$

- 위의 코드북에서 <DEPTH_START>, … , <DEPTH_END> 으로 deterministic 한 mapping 을 한다.
  이는 정답 label 이 되며, 이를 goal 로 해서 MolmoAct 의 Space Perception token 을 학습시킨다.

$$
f_{\theta}^{\text{MolmoAct}}:\ \mathcal{I} \to \mathcal{D},
\qquad d = f_{\theta}^{\text{MolmoAct}}(I)
$$

---
{% include figure
   image_path="/assets/images/robot/DPtoken.png"
   alt="DPtoken"
   caption="Depth perception token 출력 예시"
   class="align-center"
%}


#### 2.5.3 2D Trajectory token 생성  

trajectory 는 intermediate representation 으로 기능한다. EE 의 2D trajectory 를 생성하고, 이 trajectory 를 다음 행동 명령과 함께 예측하도록 학습한다.

 {% include figure
   image_path="/assets/images/robot/molmo_traj.png"
   alt="molmo_traj"
   caption="Trajectory 를 화면에 Overlay 한 예시"
   class="align-center"
%}

#### 2.5.4 전체 생성 파이프라인

$$
p(d, \tau, a \mid I, T)
=
\prod_{i=1}^{M+2} p(d_i \mid I, T, d_{<i})
\;\times\;
\prod_{j=1}^{L} p(\tau_j \mid I, T, d, \tau_{<j})
\;\times\;
\prod_{k=1}^{D} p(a_k \mid I, T, d, \tau, a_{<k}).
$$

depth perception token, trajectory token, action token 을 각각 autoregressive 하게 생성함.  

{% include figure
   image_path="/assets/images/robot/molmo_architecture.png"
   alt="MolmoAct inference pipeline"
   caption="MolmoAct inference pipeline"
   class="align-center"
%}  


## 3. Experiments  

MolmoAct의 실험 구성은 아래 4가지.

1) 사전학습 직후 zero-shot generalization 성능  
2) 훈련 분포 근처에서의 OOD robustness  
3) Visual reasoning trace를 통한 steerability  
4) MolmoAct Dataset을 이용한 mid-training 효과  

SOTA 성능 개선을 보이는 실험이 많지 않은 듯 하다(?)

---

### 3.1 Main Manipulation Performance (Zero-shot Generalization)

#### 3.1.1 Evaluation Setup  

사전학습 직후의 **out-of-the-box generalization** 성능을 평가
- Benchmark: SimplerEnv visual-matching benchmark  

- 플랫폼: WidowX, Google Robot  

- 과업 유형: visual matching 기반 manipulation task  

- 목적:
  - task-specific fine-tuning 없이
  - 사전학습만으로 즉시 실행 가능한 generalist policy인지 평가  

MolmoAct-7B-D-PRETRAIN은 다음의 generalist policy들과 비교된다.

- TraceVLA  
- RT-1X  
- OpenVLA  
- RoboVLM  
- Emma-x  
- π₀, π₀-FAST  
- Octo  
- Magma  
- HPT  
- SpatialVLA  
- GR00T N1.5  

대부분의 baseline은 zero-shot setting에서 평가

---

#### 3.1.2 Evaluation Metric  

- **Task Success Rate (%)**
  - task를 완전히 성공적으로 완료했는지에 대한 binary metric

---

#### 3.1.3 Results  

MolmoAct-7B-D-PRETRAIN은 SimplerEnv benchmark에서  
**70.5% zero-shot success rate**를 달성하였다.

이는 GR00T N1.5, π₀, π₀-FAST, Magma를 포함한  
대부분의 기존 generalist VLA 모델을 상회하는 성능이다.

또한 동일한 RT-1 subset(OXE)으로 fine-tuning을 수행한 경우,  
성능은 **71.6%**까지 향상되며  
Magma 대비 **+3.2%p** 우위를 보인다 (Table 1).

이 결과는 MolmoAct가:
- 강력한 zero-shot generalist이며
- fine-tuning을 위한 매우 효과적인 initialization임을 시사한다.

{% include figure
   image_path="/assets/images/robot/table1.png"
   alt="table1"
   caption="table1"
   class="table1"
%} 

---

### 3.2 Out-of-Distribution Robustness (Generalization Stress Test)

#### 3.2.1 Evaluation Setup  

훈련 분포 근처에서의 **robustness 및 OOD generalization**을 평가하기 위해  
다양한 변형 조건을 포함한 generalization test를 수행한다 (Section 5.3).

평가 조건은 다음과 같다.

- In-distribution  
- Language variations  
- Spatial variations  
- Distractors  
- Novel objects  

이 실험은 **zero-shot 성능 평가가 아니라**,  
fine-tuned policy가 **환경 변형에 얼마나 안정적인지**를 보기 위한 것이다.

---

#### 3.2.2 Evaluation Metric  

- **Task Progression**
  - 사전에 정의된 중간 단계(milestone)를 얼마나 달성했는지를 점수화한 metric. 즉, task 를 작은 단계로 분해한 것.

task progression 예시는 아래와 같음.

- 0.25: 초기 핵심 단계 달성 (예: grasp)
- 0.5 / 0.75: 중간 단계
- 1.0: task 완전 성공  

(progress definition은 Appendix에 있음.)

---

#### 3.2.3 Results  

MolmoAct는 모든 OOD 조건에서  
OpenVLA, π₀-FAST 대비 **일관되게 높은 task progression**을 기록한다.

특히:
- spatial variation
- distractors
- novel objects  

조건에서 성능 격차가 크게 나타나며,  
이는 MolmoAct의 structured action reasoning이  
훈련 분포 근처의 변형에 대해 더 안정적임을 보여준다.

{% include figure
   image_path="/assets/images/robot/figure6a.png"
   alt="figure 6a"
   caption="OOD Task progression"
   class="OOD Task progression"
%}

---

### 3.3 Effect of MolmoAct Dataset (Mid-training)

#### 3.3.1 Evaluation Setup  

MolmoAct Dataset(curated action reasoning data)을 사용한  
**mid-training의 효과**를 평가한다 (Section 5.3).

실제 로봇 과업을 대상으로,
- MolmoAct (dataset 사용)
- MolmoAct (dataset 미사용)
- π₀-FAST
- OpenVLA

를 비교한다.

대상 task:
- Close Lid  
- Rotate Pot  
- Pour Tea  

---

#### 3.3.2 Evaluation Metric  

- **Task Progression** 

---

#### 3.3.3 Results  

MolmoAct Dataset으로 mid-training을 수행한 경우,
task progression 이 향상됨.

즉,
- **action reasoning 중심으로 정제된 데이터**가
  모델 학습에 효과적임.

{% include figure
   image_path="/assets/images/robot/figure6b.png"
   alt="figure 6b"
   caption="Task progression under mid-training"
   class="figure6b"
%}

---

### 3.4 Steerability via Visual Reasoning Traces

#### 3.4.1 Experimental Setup  

MolmoAct는 사용자가 제공하는  
**2D visual reasoning trace (trajectory)**를 조건으로 받아  
행동을 조종(steer)할 수 있음.

(Teleop의 새로운 방식 정도인 것 같은데..?)

---

#### 3.4.2 Results (Qualitative & Quantitative)

- 동일한 scene 및 instruction에서도
- 제공되는 trace에 따라
- 전혀 다른 manipulation 행동이 생성됨

Trace segment 수가 증가할수록
task 성공률 및 progression이 향상됨.

이며, **test-time conditioning 입력**에 해당한다.

{% include figure
   image_path="/assets/images/robot/figure7.png"
   alt="table1"
   caption="test-time instruction"
   class="table1"
%}

---

## 4. Key Contributions

- BPE 를 고려한 Action bin mapping. 즉, 기존의 VLA 보다 더 개선된 action token 할당.  

- Depth perception token 을 auxiliary token 으로 도입. 기존의 LLM 이 학습한 자연어 token 과 별개로 공간을 기술할 수 있는 언어를 만듦.  

## 5. Limitations  

- Hardware embodiment 사이의 generalization 은 실험되지 않음. Google Robot, Widow X robotic Arm을 사용함.  

- camera view 변화, calibration 변화 등에 robust 하지 않음. Generalization 이 아님

- Depth token 이 물리적 metric 이 반영된 것이 아니라 collision 등을 반영하지는 못 하는 것 같고, 그냥 이미지를 더 잘 이해할 수 있게 만든다 정도 아닌가?   

- Depth camera 를 쓰지 않은 이유는 VLM 사전학습 데이터와 맞지 않아서. 그러면 근본적으로 VLM 기반으로 VLA 만드는 건 한계가 있지 않을까?  

- action token 이 joint command 인데, 이러면 근본적으로 embodiment generalization 이 안 되지 않나?
