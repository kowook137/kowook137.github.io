---
title: "Boston dynamics 와 Toyota 협업업"
date: 2025-09-02
tags: [LBM, Humanoid, GPU, VLA, Toyota_Ri]
toc: true
---
# 하나의 모델로 하부까지 구동 가능한 휴머노이드 제어

https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/
에서 보면 사람이 움직이는 대로 휴머노이드 로봇이 움직인다.

작업을 수행할 때는, 상부의 manipulation 과 동시에 하부의 무릎도 움직인다.
어떻게 한 거지?

boston dynamics 는 기존에 mpc 에 강한 회사였다고 함.

여기서 LBM 을 처음 들었다.
Large Behavior model 이라는 뜻으로, Toyota 에서 강함.

# 이미지
{% include figure
   image_path="/assets/images/robot/diffusion-transformer.png"
   alt="Diffusion Transformer pipeline"
   caption="Diffusion Transformer로 생성된 액션과 액션 스페이스 개요"
   class="align-center"
%}

# 질문사항
1. LBM 은 VLA 와 어떻게 다른가?
2. 어떻게 하부까지 제어되는가?
3. MPC 같은 dynamics 기반 제어와 LBM 기반 제어가 어떻게 융합되었는가?


