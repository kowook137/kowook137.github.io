---
title: "Simulation 툴 정리"
date: 2025-12-26
tags: [Dynamics, robotics, Simulation, RL]
toc: true
mathjax: true
---

## 1. Mujoco
### 1.1 기본 개요
Reinforcement Learning 파이프라인에 가장 기본적으로 활용되는 툴. 가볍고 빠르며, 동시에 RL 돌리기 좋다.
### 1.2 특징
접촉이 많은 휴머노이드 손, 발 등에 사용. 변형 해석을 하지 않고, contact, friction 등의 제한을 optimization 하나에 묶어서 푼다.

### 1.3 링크
> https://github.com/google-deepmind/mujoco
## 2. Drake
### 2.1 기본 개요
Drake는 모델링, 시뮬레이션, 최적화, 제어까지 다 할 수 있는 하나의 툴 킷.
### 2.2 특징, 사용 예시
왜 넘어지는지 수학적으로 규정하고, 보행 한 스텝을 최적화로 설계하고, MPC/WBC 구조를 만들고, 접촉력/모멘트를 해석
### 2.3 링크
> https://drake.mit.edu/
## 3. Move it
### 3.1 기본 개요
ROS 생태계 기반 툴
### 3.2 사용 예시
운영 lifetime 에 사용.