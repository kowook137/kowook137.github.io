---
title: "AI from Scratch"
date: 2025-01-03
tags: [AI, Robot, Physical AI]
toc: true
---

## 1. 로봇에서 현대 AI 의 흐름
 (1) World 는 연속적, 구조적 Manifold 이다.  

 (2) 관측은 불완전하고 고차원이다. (Vision, Proprioception)  

 (3) 표현(latent representation)이 필요하다. (encoder) 

 (4) 그 위에서 시간적, 확률적 변화를 모델링한다.(Transformer/Diffusion/Flow)  
 
 (5) 그 결과를 제어(Action)으로 연결한다.

 ## 2. Line By Line

 ### (1) World 는 연속적, 구조적 Manifold 이다.

현실 세계의 상태는 ℝⁿ 전체를 차지하지 않는다.
실제로는 강한 제약을 가진 연속적인 곡면(manifold) 위에 있다.
예를 들어, 로봇의 Joint Angle 은 임의의 각도가 아니라 구조적으로 제약된 공간.  
Contact 또한 Physically feasible 한 Contact 만 존재한다.

### (2) 관측은 불완전하고 고차원이다.  

로봇이 보는 것은 RGB, IMU, Depth, Joint Encoder 등만 본다.  
즉, 관측 $o$ 는 상태 $x$ 의 일부의 projection.
$$
o = h(x) + noise
$$

### (3) latent representation이 필요하다.

고차원 관측을 의미 있는 저차원 공간(latent) 로 mapping  

- geometry 를 보존하면서  
- dynamics 가 단순해지도록  
- 제어에 필요한 정보만 남긴다.  

좋은 representation 이란  
- time 에 따라서 smooth  
- action 에 대해서 locally linear  

### (4) 그 위에서 시간적, 확률적 변화를 모델링함.

여기서 sequential 하게 trajectory 를 생성하거나, physically feasible 하도록 manifold 상에서 action 을 생성할 수 있도록 Diffusion, Flow matching 등이 등장함.  
위의 것들은  
$$
p(future|past, goal) \rightarrow p(action/trajectory)
$$

즉, distribution &rarr; distribution 의 Mapping을 구현하는 방법의 차이.

### (5) Action 으로 출력함.

여기서 RL, Model Based Control 등을 다룬다.
