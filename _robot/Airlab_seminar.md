---
title: "AIRLab Seminar"
date: 2025-12-30
tags: [Robot]
toc: true
---

## 1. Task Planning, BRL corrective feedback
Long-time Horizon 에서 Task Planning 의 어려움을 해결하기 위해 추상화 한 후에 Plan 하고, 이를 다시 refine 하는 과정을 통해 해결한다.
추상화 하는 과정에서의 대표변수 선정이 중요함.

큰 주제는 Task Planning(?)
BRL corrective feedback.

## 2. 회사에서 어떻게 프로젝트를 진행하는가?
산출물: 상세설계서, 코드, 인터페이스 명세서, 기능 정의서, 시스템 구성도
코드 배포는 GitLab CI/CD 를 활용해서.

DevContainer는 VS Code 설정을 컨테이너로 이식할 수 있음.
도커의 개발 환경 버전. 어떤 PC 에서는 되고, 어떤 PC 에서는 안 되는 일이 사라짐.

## 3. VLN+Dataset, Server + Docker

### Server 사용 법
1. 유저 생성(원종, 장현)
2. Docker 권한 받기 (원종 or 장현)
3. Docker 기본 container 구축
4. 필요 시 docker 내부에 container 설치
5. 기본 명령어 (docker start wook, docker exec -it woojin bash)

## 4. Multi Agent System

> Magnnet: Multi-agent graph neural network-based efficient task allocation for autonomous vehicles with deep reinforcement learning.

Thesis: Multi agent Task allocation.

## 5. 사용할 만한 Tool 소개

### 5.1 로봇 Trajectory 관련
Motion Comparator: 로봇 궤적 시각화
Host 버전 사용. collision checking 같은 heavy 한 기능은 없음.
Quaternian 등이 어떻게 움직이는지 나타낼 수 있는 그래프 표출 기능도 있음.

### 5.2 Zotero - 논문 정리 Tool
주제별 논문을 정리할 수 있음.
PDF reader 가 괜찮아서 필기, 하이라트 잘 되고, Zotero Connector 라는 Plugin 도 존재해서
Chrome 상에서 읽다가 바로 넣을 수 있음. 동기화 기능도 제공됨.

## 6. VLA 에 대한 회의?

VLA는 결국 LLM 성능에 얹혀가는 것인 듯 하다.
Environment 과 상호작용하는 것이 아닌 이상에는 제대로 된 학습이 될 수 없다.
이 해결책으로 World Model 이 될 수 있다.
Veo3 가 좋았던 것이 이전의 상호작용을 Consistency 있게 유지할 수 있어서 좋았다.

DayDreamer - World Model 개념이 적용되어서 학습한 것으로, 훨씬 잘 학습하더라.

Self Distillation, V-JEPA 2

사람도 뇌 안에 각자의 World Model 이 있고, 이를 실제 World 와 차이를 줄이며 학습을 한다.

## 7. HEART (2025), RA letters

LLM-based Robot Planning

## 8. CJ 대한통운 TES LAB Track 인턴십

## 9. CARE(Collision ~~), ViNT, VLN

# 10. 임베디드 보드 설계

TTL: Transistor to Transistor Logic
Autodesk Fusion. 이거 설계하는 거 공부해야 함.

(1) 부품 선정
(2) 회로도 작성
(3) pcb 레이아웃 설계