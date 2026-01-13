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

### 2.3 학습 데이터  

OXE, BC-Z, V2, RT-1, Molmo ACT dataset (curated)  

### 2.4 GPU hours  

9,216 GPU hours, GR00T 대비 5배 절감

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


 (1) Teacher (Depth tokenization)

\[
\text{Model} \equiv \text{teacher model}
\]

\[
D \in \mathbb{R}^{H \times W}
\quad (\text{dense depth map})
\]

\[
\mathrm{VQ\text{-}VAE} : D \mapsto z^{\text{depth}}
\]

\[
z^{\text{depth}} = (z_1, \dots, z_L),
\qquad z_i \in \{1, \dots, 128\}
\]

\[
\text{Deterministic encoding: }
z^{\text{depth}} \mapsto d
\]

\[
d = \langle \text{DEPTH\_START} \rangle,\;
\langle \text{DEPTH\_}z_1 \rangle,\;
\langle \text{DEPTH\_}z_2 \rangle,\;
\dots,\;
\langle \text{DEPTH\_END} \rangle
\]

---

  (2) MolmoAct (Distillation target)

\[
(I, a) \xmapsto{\text{MolmoAct}} \hat{d}^{\text{depth}}
\]


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







## 실험 결과

## 핵심 contribution 및 기존 논문들과의 차이

## 한계점/개선 사항  
