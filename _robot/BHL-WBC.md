---
title: "Berkeley Humanoid lite"
date: 2026-01-07
tags: [Humanoid, Whole Body control]
toc: true
---

## Berkeley Humanoid Lite 논문 정리

#### 논문 초점

휴머노이드 로봇을 연구할 수 있는 저가형 플랫폼을 만들었다. 하드웨어 설계에 주로 초점이 있음.

{% include figure
   image_path="/assets/images/robot/Berkeley_lite.png"
   alt="Berkeley humanoid lite"
   caption="Berkeley Humanoid lite"
   class="align-center"
%}  

위의 이미지가 Calibration 을 위한 자세임.

---

#### Locomotion 관련 사항

- Isaac Gym-trained Policy 로 zero-shot 으로 Locomotion 을 수행할 수 있음.  

- Locomotion 을 위한 policy 는 MLP 로 구성하여 E2E 로 동작함.  

- 보행 문제 formulation 은 POMDP, 학습 방법은 PPO  

- observation input  
(1) Base angular velocity: 몸통이 얼마나 빨리 회전/비틀림 하는지  
(2) Projected gravity vector: 몸통이 얼마나 기울어져 있는지.(roll, pitch 정보)  
(3) Joint positions, velocity: 현재 자세, joint 속도  

- 학습 target  
MLP 로 $(\text{state}, \text{goal}) \mapsto \text{joint command}$ 를 학습함.

- inference Rate  
MLP 로 근사된 policy 는 25Hz 로 돌아감. MBC 사용 X.  

---

#### Hardware 관련 사항  

- 사용 모터  
M6C12 BLDC Drone Motor, 5010 BLDC Drone motor.  
모두 MAD 모터 사용.  
- Gearbox  
사용된 Gearbox 는 3d printed, Cycloidal Gearbox.  

{% include figure
   image_path="/assets/images/robot/Gearbox.png"
   alt="Cycloidal Gearbox"
   caption="Cycloidal Gearbox"
   class="align-center"
%}

---

#### Future Works, Limitations

 - Gearbox 가 3D-printed 인데, BLDC 모터의 열로 인해서 Mechanical property 가 보장되지 않음.  

 - 위의 이유로 Long-Time manipulation, locomotion 에 한계 존재.
 ---

#### 배운 것들

- MLP 가 빠른 추론이 필요한 곳에는 여전히 많이 잘 쓰이고 있다.  

- MLP 같은 것들은 BHL 처럼 Intel N95 mini PC 에서도 inference 충분히 가능하다. GPU 를 쓸지 말지, 고려해서 선택해야지 무조건 GPU 있는 칩이 좋은 건 아님.  

- 빠른 제어는 무조건 Model Base control 인 줄 알았는데, 그게 아니다. 여기처럼 MLP 로 일종의 반사신경을 구현해서 locomotion 만 구현하려면, MLP 로 충분하다.  

- BHL 도 그렇고, RL 로 학습된 locomotion 은 보통 앞으로 이동하기 위해 무게중심을 이동시키면서, 동시에 안 넘어지려고 버티는 "Controlled falling" 인 것 같다. 사람처럼 planning 기반으로 걷는 것과는 다름.

- Proprioceptive actuator  
모터임과 동시에 센서. 전류로 토크 추정, 위치는 encoder, 위치의 미분으로 속도까지. 빠른 고대역폭 제어를 가능하게 함.  

- Actuator test  
아래 그림과 같이 구성하여 성능 테스트를 하네. 알아두면 좋을 듯.  

{% include figure
   image_path="/assets/images/robot/Actuator test.png"
   alt="Actuator test"
   caption="Test method for actuator"
   class="align-center"
%}
