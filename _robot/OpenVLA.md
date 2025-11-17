---
title: "Open VLA"
date: 2025-11-17
tags: [Robot, Humanoid]
toc: true
---

# OpenVLA 논문 상세 요약 (Markdown)

## 1. 개요

OpenVLA는 **7B 파라미터 규모의 오픈소스 Vision-Language-Action(VLA)** 모델로, 로봇 조작(Manipulation) 작업에서 강력한 범용성(generalization)을 보이는 정책(policy) 모델이다. 기존 구글 RT-2-X와 같은 VLA 모델은 폐쇄적이거나 파인튜닝이 어렵다는 한계가 있었지만, OpenVLA는 **모델 가중치, 학습 코드, 데이터 파이프라인을 모두 공개**하여 누구나 재현 및 확장이 가능하도록 설계되었다.

OpenVLA는 다음을 목표로 한다:

* 멀티 로봇에 대해 바로 사용할 수 있는 일반 목적 로봇 정책
* 적은 데이터로 새로운 환경으로 빠르게 fine-tuning
* 저사양 GPU에서도 LoRA 및 quantization을 통한 효율적 학습 및 추론

---

## 2. 모델 구조

OpenVLA는 다음 3가지 요소로 구성된다:

### 2.1 Vision Encoder

* **DINOv2**와 **SigLIP**을 동시에 사용하여

  * SigLIP → 인터넷 규모 시각-언어 의미 이해 능력
  * DINOv2 → 정밀한 공간 정보(spatial reasoning)
* 두 시각 특징을 채널 방향으로 concat하여 사용

### 2.2 Projector

* Vision encoder 출력(feature)을 **언어 모델(LLM)의 embedding 공간**으로 투영하는 2-layer MLP

### 2.3 LLM Backbone

* **Llama 2 7B** 기반의 대형 언어 모델
* 이미지 패치와 명령어를 입력받아 **로봇의 7차원 제어 명령(Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)**을 예측하도록 fine-tuning

---

## 3. Action Tokenization 방식

Continuous action을 LLM이 생성하도록 하기 위해 다음과 같은 과정을 사용:

* 각 action dimension을 **256개의 discrete bin**으로 양자화(discretization)
* Llama tokenizer의 사용 빈도 가장 낮은 256개의 토큰을 action 토큰으로 재정의
* Next-token prediction 방식으로 학습

이 방식으로 VLM이 자연스럽게 로봇 제어 명령을 생성하게 됨.

---

## 4. 학습 데이터 구성

OpenVLA는 총 **970,000개의 실로봇 조작 demonstration**으로 학습되며, 주로 다음을 포함한다:

### 4.1 Open X-Embodiment Dataset 기반

* 70개 이상의 서로 다른 로봇 데이터셋
* WidowX, UR5, Franka 등 다양한 로봇
* 다양한 환경, 조작 종류, 물체 형태 포함

### 4.2 Dataset Curation

* 최소 한 개의 3인칭 카메라 보유한 manipulation 데이터만 사용
* Octo 데이터 믹스(weight)를 기반으로 균형 조정
* DROID 데이터는 diversity가 커서 일부 구간만 제한적으로 사용

이를 통해 multi-embodiment generalization 능력 강화.

---

## 5. 주요 디자인 선택(Design Decisions)

OpenVLA 개발 과정에서 실험으로 검증한 주요 결론:

### 5.1 Vision Encoder는 반드시 Fine-tuning 필요

* 기존 VLM 연구는 vision encoder를 freeze하는 것이 안정적
* 그러나 로봇 제어에서는 **정밀한 위치 판단**이 필요하므로 unfreeze가 성능 향상에 중요

### 5.2 높은 해상도(384×384)는 필요하지 않음

* 224×224와 성능 차이 거의 없음
* 384는 학습 비용이 3배 증가 → 비효율적

### 5.3 고정 Learning Rate (2e-5)가 가장 안정적

* warmup 필요 없음
* VLM 학습에서 사용한 동일 LR 유지

### 5.4 많은 Epoch 반복이 필요(총 27 epoch)

* 일반 LLM/VLM 학습은 대규모 데이터로 1~2 epochs면 충분
* 그러나 로봇 데이터는 다양성이 적기 때문에 **반복 학습이 매우 중요**

---

## 6. 주요 성능 평가 결과

OpenVLA는 기존 모델들과 비교하여 매우 높은 성능을 보였다.

### 6.1 WidowX (BridgeData V2) 평가

* RT-2-X(55B)보다 **16.5%p 높은 성공률**
* 다양한 generalization task에서 SOTA 달성

  * unseen backgrounds
  * unseen object colors
  * unseen shapes
  * unseen instructions

### 6.2 Google Robot 평가

* RT-2-X와 거의 동등한 성능
* RT-1-X, Octo보다 월등히 우수

### 6.3 Language Grounding

* 다중 객체 환경에서 명령어 기반 타겟 식별 능력이 가장 우수한 축

---

## 7. Fine-Tuning 성능

OpenVLA를 Franka 로봇에 적은 데이터(10~150 demonstrations)만으로 fine-tune 시도.

### 비교 대상

* **Diffusion Policy** (from scratch)
* **Octo**
* **OpenVLA(scratch)** (프리트레인 없이 학습)

### 결과

* 단일 명령어 기반의 단순 작업: Diffusion Policy가 여전히 강함
* **다양한 언어 지시 기반 복합 작업**: OpenVLA가 압도적으로 우수
* OpenVLA는 **모든 작업에서 50% 이상의 성공률**을 최초로 달성한 일반 모델

이는 “대규모 robot + VLM pretraining”의 효과를 정량적으로 보여줌.

---

## 8. Parameter-efficient Fine-Tuning (LoRA)

전체 fine-tuning은 매우 무겁기 때문에 효율적 방법을 실험:

| 방식              | 성능              | 파라미터 수         | GPU 메모리   |
| --------------- | --------------- | -------------- | --------- |
| Full FT         | 최고              | 7.1B           | 163 GB    |
| Last layer only | 낮음              | 465M           | 51 GB     |
| Frozen vision   | 낮음              | 6.7B           | 156 GB    |
| Sandwich FT     | 중간              | 914M           | 64 GB     |
| **LoRA r=32**   | **Full FT에 근접** | **97M (1.4%)** | **59 GB** |

결론: **LoRA (rank=32)**가 가장 효율적 → 1 GPU(A100)에서 10~15시간이면 fine-tuning 가능.

---

## 9. Quantization을 통한 추론 최적화

OpenVLA는 7B 모델이라 16GB GPU에서 추론이 빡빡하지만, 다음으로 해결 가능:

### 9.1 bfloat16 (기본)

* 약 16.8GB 메모리 필요

### 9.2 int8

* 속도가 크게 느려져 실제 로봇 제어에서는 성능 저하 발생

### 9.3 **int4 quantization** (가장 추천)

* 메모리: 7GB
* 속도: bfloat16과 비슷하거나 더 빠름
* 로봇 실제 제어에서도 성능 유지

즉, 일반 PC의 RTX GPU에서도 OpenVLA를 활용할 수 있음.

---

## 10. 한계점 및 향후 과제

OpenVLA는 강력하지만 몇 가지 개선 여지가 있음:

### 10.1 단일 이미지 입력만 지원

* 이력(history)이나 proprioception 부재
* 멀티 카메라 환경 지원이 필요

### 10.2 추론 속도 향상 필요

* 고주파수 제어(예: 50 Hz ALOHA)에서는 아직 사용 불가능

### 10.3 90% 이상의 안정적 성공률 달성은 앞으로의 과제

* 안정성이 중요한 실제 산업 수준에는 아직 부족

### 10.4 VLA 설계 원리에 대한 탐구 필요

* 비전 백본 크기 변화
* 인터넷 데이터 + 로봇 데이터의 co-training 효과
* 시각 특징의 종류와 적절한 융합 방식

---

## 11. 결론

OpenVLA는 다음을 동시에 만족하는 최초의 범용 로봇 조작 모델이다:

* **완전 오픈소스** (모델 가중치 + 코드 + 학습 파이프라인 공개)
* **멀티 로봇 제어 가능**
* **소량의 데이터로 강력한 fine-tuning 성능**
* **LoRA, int4 기반 효율적 학습 및 추론 지원**

이는 앞으로 VLA 기반 실제 로봇 조작 연구의 대중화 가능성을 크게 확대한다. 앞으로의 VLA 연구를 위한 튼튼한 ‘기반 모델’이 될 것으로 기대된다.

(본 요약은 업로드된 논문 내용을 기반으로 작성됨.)
