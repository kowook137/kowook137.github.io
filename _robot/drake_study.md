# Drake 구조 정리 (Flow 중심 관점)

## 1. Drake를 바라보는 핵심 관점
 
수학적으로 정의된 시스템(System)들의 흐름(flow)을 선언하는 프레임워크.

중요한 것은
- 어떤 신호가 어디에서 어디로 흐르느냐

---

## 2. Drake 프로그램의 전체 구조

Drake 프로그램은 항상 다음 구조를 따른다.

(1) System 들을 정의함.  

(2) System 들을 연결하여 Diagram 생성  

(3) Simulator 가 Diagram 전체를 시간 적분

## 3. System in Drake  

System (추상 개념) |
--- 
├── LeafSystem (사용자 정의 블록)
├── MultibodyPlant (동역학 엔진)
├── SceneGraph (기하 / 시각화 관리)
└── 기타 System 들   

System은 다음을 가질 수 있다.

- Input Port
- Output Port
- Internal State (optional)  

## 4. LeafSystem의 역할

LeafSystem은 Drake에서 **사용자가 직접 정의하는 계산 블록**이다.

- 제어기
- 필터
- 추정기
- 신호 처리 블록

LeafSystem은 다음 질문에 답하도록 설계된다.

- 어떤 입력을 받는가?
- 어떤 출력을 내보내는가?
- (내부 상태는 무엇인가?)  

## Multibody Plant  

MultibodyPlant는 Drake에서 **물리 세계 자체를 담당하는 System**이다.

- generalized position (q)
- generalized velocity (v)
- dynamics equation
- contact / constraint
- actuator 모델

MultibodyPlant는 다음 mapping을 수행한다.

```
(state, input) -> (state_dot)
```  

## 6. Diagram: System을 연결한 그래프

Diagram은 여러 System을 **포트 단위로 연결한 그래프**이다.  

Controller (LeafSystem)
        │  (actuation / torque command)
        ▼
MultibodyPlant
        │  (state: q, v)
        ▼
Controller

## 작업 흐름
Drake에서 흔한 작업 흐름은 다음과 같다.

1. URDF / SDF 로 로봇 모델 정의
2. MultibodyPlant 로 dynamics 생성
3. LeafSystem 으로 제어기 작성
4. Diagram 으로 연결
5. Simulator 로 검증


즉, ROS2 랑 비슷하게 Leaf system(노드) 를 구성하고, 이들을 서로 연결하고, 이를 통해 전체 제어 루프 시스템을 구성한다.