---
title: "로봇에 사용되는 통신 방식"
date: 2025-11-17
tags: [Robot, communication]
toc: true
---

# ROS vs DDS vs Xenomai Shared Memory

---

## 1. 계층 관점 비교

```
[ Application / Control Logic ]
        |
        |  (Framework API)
        v
[ ROS 2 (rclcpp / rclpy) ]
        |
        |  (Middleware Abstraction)
        v
[ DDS (Cyclone DDS, Fast DDS) ]
        |
        |  (RTPS Wire Protocol)
        v
[ UDP / Shared Memory ]
        |
        |  (Kernel / RT)
        v
[ Xenomai RT Tasks + Shared Memory ]
```

---

## 2. 각 기술의 정체성

### ROS (ROS 2)

* 성격: **로봇 소프트웨어 프레임워크**
* 목적:

  * 노드 관리
  * 토픽 / 서비스 / 액션
  * 파라미터 서버
  * TF (좌표계 관리)
  * 런치 시스템
  * 로깅 및 디버깅
* 내부 통신 엔진: DDS

> 통신뿐 아니라 로봇 애플리케이션 개발 전반을 다룸

---

### DDS (Cyclone DDS 등)

* 성격: **통신 미들웨어**
* 목적:

  * Publish–Subscribe
  * Data-Centric 통신
  * QoS 기반 계약형 통신
* 특징:

  * 브로커 없음
  * 자동 discovery
  * IDL 기반 타입 계약
  * 분산 시스템 지향

> “누가 누구에게”가 아니라 “어떤 데이터가 공유되는가”

---

### Xenomai + Shared Memory

* 성격: **실시간 실행 환경**
* 목적:

  * Hard real-time 보장
  * μs 단위 jitter 억제
* 특징:

  * RT 커널 / 코커널 구조
  * RT task 간 메모리 직접 공유
  * 메시지 / 통신 개념 제거

> 통신이 아니라 “같은 메모리를 본다”

---

## 3. 통신 / 실행 방식 비교

### ROS

```
DDS 수신
 → rmw
   → rcl
     → Executor
       → Callback Queue
         → User Callback
```

* 콜백 스케줄링
* 큐잉
* 타입 추상화
* 편의성 중심 설계

---

### DDS Direct

```
RTPS
 → DDS DataReader
   → read() / take()
     → User Code
```

* waitset 기반 동기화 가능
* QoS 세밀 제어
* executor 없음

---

### Xenomai Shared Memory

```
[ RT Task A ]
    |
 shared struct
    |
[ RT Task B ]
```

* 메시지 없음
* serialization 없음
* 복사 없음
* lock-free / atomic 접근

---

## 4. Latency / Jitter 발생 요인

### ROS에서 발생 가능한 지연 원인

* Executor 스케줄링
* Callback 큐 대기
* 다중 토픽 간 priority inversion
* 타입 변환 / 메모리 복사
* Python 사용 시 GIL 영향

---

### DDS Direct에서의 지연

* RTPS 전송 지연
* QoS 정책 (History, Retry 등)
* 네트워크 스택

---

### Xenomai Shared Memory에서의 지연

* 캐시 / 메모리 접근 수준
* 사실상 통신 지연 없음

---

## 5. 기능별 비교 표

| 항목            | ROS 2   | DDS Direct | Xenomai SHM  |
| ------------- | ------- | ---------- | ------------ |
| 성격            | 프레임워크   | 미들웨어       | RT 실행 모델     |
| 통신 모델         | Pub-Sub | Pub-Sub    | 메모리 공유       |
| 타입 정의         | .msg    | .idl       | C/C++ struct |
| QoS           | 추상화 제공  | 원형 그대로     | 없음           |
| Discovery     | 자동      | 자동         | 없음           |
| Serialization | 있음      | 있음         | 없음           |
| Message Copy  | 있음      | 최소화 가능     | 없음           |
| Latency       | ms ~    | 수십 μs ~    | 수 μs 이하      |
| Jitter        | 있음      | 낮음         | 거의 없음        |
| 분산성           | 높음      | 매우 높음      | 없음           |
| 디버깅           | 매우 좋음   | 보통         | 매우 어려움       |
| 구조 변경         | 쉬움      | 보통         | 매우 어려움       |

---

## 6. 제어 주기 기준 선택 가이드

* ≤ 200 Hz, 연구 / 개발 / 디버깅 중심 → ROS
* 200–1000 Hz, 저지연 필요 → DDS Direct
* ≥ 1 kHz, hard real-time → Xenomai + Shared Memory

---

## 7. 휴머노이드에서의 실제 사용 패턴

```
[ Xenomai RT Core ]
 - Motor Control
 - Estimator
 - WBC
        |
  (Shared Memory Boundary)
        |
[ Linux User Space ]
 - DDS Direct
 - ROS 2
 - UI / Logging / Planning
```

* 저수준 제어: SHM / DDS
* 고수준 로직: ROS

---

## 8. 산업용 SDK / 휴머노이드 SDK의 선택 해석

* ROS 미사용 ≠ DDS 미사용
* ROS 제거 = 프레임워크 오버헤드 제거
* DDS Direct = 저지연 + 분산 + 구조적 명확성

---

## 9. 최종 정리 문장

ROS는 DDS를 사용하지만, DDS는 ROS가 아니다.

휴머노이드 제어에서는 통신 편의성보다 결정성과 지연 한계가 먼저다.

