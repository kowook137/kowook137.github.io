---
title: "로보티즈 오픈랩 투어"
date: 2026-01-16
tags: [Robot]
toc: true
---

# 1. Robotics 구분  

bipedal, quadruped, mobile manipulator, hand, humanoid 의 크게 다섯가지.  

OMY 에는 Friction Compensation, Gravity compensation, Spring Effect, Collision 구현 등의 기능이 있네  

AI Worker 는 Jetson Orin 으로 Training, Inference 둘 다 가능하다. 

OMX, OMY 는 소스가 다 github 에 있으니까, 중력 보상, 마찰 보상 관련 코드를 참고하면 괜찮을 듯 하다.  

AI worker 는 Jazzy. Dynamixel Hardware Interface 에서 Dynamixel을 ROS2 에서 제어할 수 있도록 함.  

Atlas 로봇 teleoperation 할 때는 teleop 과 무게중심 차이 문제 해결을 위해 MPC 사용함.

VLA 만으로는 쉽지 않고, Rule base 와 결합해서 task 실행했다고 하는데 정확히 어떻게 결합했다는 거?  

Isaac Sim 돌릴 때 lab 실에 있는 책상 환경 스캔해서 그걸 시뮬레이션에 그대로 구현해서 학습함. -> 뭘로 스캔함? 3D 스캐너로 스캔한건가?  

붓 만 잡던데 학습 시키고 나서 object 일반화는 안 되는건가? Language 지시를 씹는 문제는 없을려나  -> 없음. object 별로 다르게 잘 잡는다. 



VLA 는 총 10시간 정도, 한 에피소드에 30초 정도로 했음.
제노 기반으로 Shared memory 로 통신 없이 구현하려고 함. -> ROS 는 사용함. Unitree 를 분석 해 봤을 때 ROS 를 썼다고 latency 가 있다고 생각하지는 않는다.

휴머노이드는 QDD 사용하는데, 위치 추종성 보다는 전류 제어로 토크 제어라서(위치 제어로 전류 제어를 하는 거라 latency 가 의미 있지는 않다.)


language 씹히는 건 language 를 잘 못 넣어서.
VLM 자체의 적중 문제이다. 데이터 셋의 분포 보다는 VLM 자체의 detection 문제에 가깝다.
붓이 아니라 빨간색의 긴 막대

GR00T 기준 Chunk 16개. inference 주기는 10Hz. 작업에 따라서 다른데 Pick and place 에서는 그렇다. 잡는 복잡한 작업은 30Hz.
실패한 데이터 모으는게 어려운데 그냥 실패할 것 같은 지점에서 다시 데이터 쌓고 한다.
보통 init 포지션에서 하는데, 시작 포지션을 다르게 해서 데이터 셋을 모은다.

실패 데이터 셋을 넣는 것도 애매한데 실패를 모방하는게 맞나?


짧으면 비동기 처리를 할 수 없어서. 

Chunk 에서 가장 끝에 가까운 지점을 잡아서 돌린다.

천천히 해야 한 액션을 여러개의 Chunk 로 쪼개서 보기 위해서 천천히 teleop 해서 데이터 셋을 쌓는다.

환경은 물류로 가정해서 수행함.

GR00T Dream 이나 Physical AI Tools 로 환경을 증강해서 학습을 시켜서 Environment generalization.

같은 장면에 다른 행동을 한다고 하면 얘가 헷갈리니까 같은 장면에서는 같은 행동을 해야 한다.
