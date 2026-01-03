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