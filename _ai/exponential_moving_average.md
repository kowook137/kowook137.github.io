---
title: "Exponential_moving_average"
date: 2026-03-09
tags: [AI, Deep Learning]
toc: true
---

# Exponential Moving Average (EMA)

## 1. 개요

Exponential Moving Average(EMA)는 딥러닝 학습 과정에서 모델 파라미터의 변동성을 줄이기 위해 사용하는 **지수이동평균 기반의 파라미터 평균화 기법**이다.  
실무에서는 보통 **현재 학습 중인 모델의 파라미터**와는 별도로, 그 파라미터들의 **시간에 따른 smoothed copy**를 유지하는 방식으로 사용한다.

일반적인 mini-batch 기반 학습에서는 각 step의 gradient가 전체 데이터에 대한 정확한 gradient가 아니라, 해당 mini-batch에 대한 noisy estimate이기 때문에 모델 파라미터가 step마다 흔들린다. EMA는 이 흔들림을 완화하여, 평가 시점에서 더 안정적이고 일반화 성능이 좋은 모델을 제공하는 경우가 많다.

특히 다음과 같은 조건에서 EMA의 효과가 크게 나타나는 경우가 많다.

- mini-batch 크기가 작아 gradient variance가 큰 경우
- 학습 후반부에 validation 성능이 step마다 흔들리는 경우
- 조합 최적화, 강화학습, sequence decision 문제처럼 objective 자체의 variance가 큰 경우
- 이미 높은 점수에 도달했지만 추가 성능 향상이 어려운 구간에서, 미세한 안정화가 성능 차이로 이어지는 경우

3D bin packing과 같은 조합 최적화 문제에서 EMA를 적용했을 때 성능 향상이 크게 나타난 것은, 단순히 우연이라기보다는 **mini-batch 학습이 유발하는 파라미터의 고주파 진동을 EMA가 완화했기 때문**으로 해석하는 것이 타당하다.

---

## 2. EMA의 기본 수식

EMA는 보통 다음과 같은 형태로 정의된다.

$$
\theta_t^{EMA} = \beta \theta_{t-1}^{EMA} + (1-\beta)\theta_t
$$

여기서 각 기호의 의미는 다음과 같다.

- $\theta_t$ : 현재 step $t$에서 optimizer에 의해 업데이트된 실제 모델 파라미터  
- $\theta_t^{EMA}$ : EMA로 유지되는 smoothed 파라미터  
- $\beta$ : decay coefficient (또는 smoothing coefficient)
  - 보통 0.99, 0.999, 0.9999 같은 값이 사용된다

이 식은 현재 파라미터 \(\theta_t\)를 그대로 사용하는 것이 아니라, 직전까지 누적된 EMA 값과 현재 파라미터를 혼합하여 새로운 EMA 파라미터를 만드는 방식이다.

이를 전개하면 다음과 같은 weighted average 형태가 된다.

$$
\theta_t^{EMA}
=
(1-\beta)\theta_t
+
(1-\beta)\beta \theta_{t-1}
+
(1-\beta)\beta^2 \theta_{t-2}
+
\cdots
$$

즉, EMA는 과거의 파라미터들을 모두 반영하되, **최근 파라미터일수록 더 큰 가중치**를 부여하는 방식이다.  
이 때문에 단순 평균보다 최근 상태를 더 잘 반영하면서도, 마지막 step 하나의 우연한 fluctuation에는 덜 민감하다.

---

## 3. EMA의 직관적 의미

mini-batch 기반 학습에서는 매 step마다 서로 다른 batch를 사용하기 때문에 gradient 방향이 조금씩 달라진다.  
따라서 파라미터는 학습 후반부에도 완전히 고정되는 것이 아니라, 최적점 주변에서 계속 진동하게 된다.

이때 마지막 step의 파라미터는 다음과 같은 한계를 가질 수 있다.

- 특정 mini-batch에 우연히 더 잘 맞춘 상태일 수 있다
- 전체 데이터 기준으로는 약간 과도하게 움직인 상태일 수 있다
- validation 또는 test 환경에서는 덜 일반화되는 방향으로 순간적으로 치우친 상태일 수 있다

EMA는 이러한 문제를 줄이기 위해, 매 step의 파라미터를 직접 쓰지 않고 그 궤적을 시간적으로 평균낸 모델을 사용한다.

직관적으로 말하면 다음과 같다.

- 원래 모델: 매 step마다 흔들리는 trajectory 상의 한 점
- EMA 모델: 그 trajectory를 부드럽게 평균낸 대표점

이 관점에서 EMA는 일종의 **temporal ensemble**로 볼 수 있다.  
즉, 여러 시점의 모델을 따로 저장해 추론 시 ensemble하는 것은 아니지만, 파라미터 수준에서 시간 방향의 평균을 취함으로써 유사한 안정화 효과를 낸다.

---

## 4. EMA가 성능을 향상시키는 이유

### 4.1 mini-batch noise 완화

EMA의 가장 직접적인 역할은 **stochastic mini-batch update로 인한 noise를 줄이는 것**이다.

mini-batch 학습에서는 실제로 사용하는 gradient가 다음과 같다.

$$
g_t = \nabla L_{B_t}(\theta_t)
$$

여기서  $B_t$ 는 step $t$에서 사용된 mini-batch이다. 
이 gradient는 전체 데이터셋에 대한 true gradient가 아니라 일부 샘플에 대한 근사치이므로, 본질적으로 noise를 포함한다.

그 결과 파라미터 업데이트는 다음과 같은 특성을 가진다.

- 매 step의 방향이 조금씩 달라진다
- 최적점 주변에서도 잔진동이 남는다
- 마지막 checkpoint 하나만으로는 전체적인 좋은 해를 대표하지 못할 수 있다

EMA는 여러 step의 파라미터를 지수적으로 평균하여 이러한 잔진동을 줄인다.  
따라서 평가 시점에서는 마지막 step보다 EMA 모델이 더 높은 성능을 보이는 경우가 많다.

---

### 4.2 일반화가 더 잘 되는 파라미터를 형성할 가능성

학습 후반부의 파라미터 trajectory를 보면, 모델은 종종 비슷하게 좋은 여러 지점 사이를 미세하게 이동한다.  
하지만 이들 중 어떤 점은 특정 batch에 우연히 더 맞춰진 상태일 수 있고, 어떤 점은 더 안정적일 수 있다.

EMA는 이 trajectory 상의 여러 지점을 혼합함으로써, 다음과 같은 특성을 가진 파라미터를 형성할 가능성이 크다.

- 더 부드러운 해
- 더 안정적인 해
- 특정 batch에 대한 과민 반응이 줄어든 해
- validation/test에서 더 robust한 해

실제로 weight averaging 계열 기법은 종종 sharp한 해보다 더 넓고 안정적인 영역을 대표하는 파라미터를 형성하는 데 도움이 된다고 해석된다.  
EMA 또한 이런 맥락에서 이해할 수 있다.

---

### 4.3 마지막 checkpoint보다 더 많은 trajectory 정보를 반영

마지막 모델은 학습 trajectory의 가장 마지막 한 지점만 반영한다.  
반면 EMA는 학습 후반부의 여러 시점에 걸친 파라미터 정보를 함께 담는다.

즉, EMA는 다음과 같은 구조적 장점을 가진다.

- 마지막 한 번의 업데이트에 덜 의존함
- 직전 수백~수천 step의 정보를 누적함
- "좋은 상태"가 잠깐 나왔다 사라져도 그 정보를 일부 보존함

따라서 학습 과정에서 이미 좋은 영역에 들어가 있었더라도, 마지막 step이 그 영역을 충분히 대표하지 못하는 경우 EMA가 더 좋은 결과를 내는 것이 가능하다.

---

## 5. 3D Bin Packing과 같은 조합 최적화 문제에서 EMA가 특히 잘 작동할 수 있는 이유

3D bin packing은 단순한 연속 회귀 문제가 아니라, **순차적 의사결정과 조합 구조를 가진 최적화 문제**이다.  
이런 문제에서는 작은 파라미터 변화가 최종 score에 비선형적 영향을 줄 수 있다.

예를 들어 다음과 같은 현상이 나타날 수 있다.

- box placement 순서가 조금만 바뀌어도 최종 utilization이 크게 달라짐
- 특정 action preference가 미세하게 흔들려도 packing quality가 바뀜
- mini-batch에 포함된 instance 분포에 따라 gradient 방향이 크게 달라질 수 있음

이런 조건에서는 학습 후반부에도 파라미터 noise가 실제 성능에 직접적인 영향을 주기 쉽다.  
즉, classification처럼 softmax confidence가 조금 달라지는 정도로 끝나는 것이 아니라, **조합 구조 전체가 달라질 수 있다**.

따라서 EMA를 통해 정책 또는 scoring model의 파라미터를 안정화하면 다음과 같은 효과가 기대된다.

- placement decision의 일관성 증가
- 특정 batch에 대한 과도한 편향 감소
- 평가 환경에서 더 robust한 packing sequence 형성
- 이미 높은 점수 영역에서도 미세한 추가 향상 가능

**추정 분산과 parameter fluctuation이 성능 병목이었던 상황**일 수 있다.

---

## 6. EMA가 평균하는 것은 무엇인가

EMA라는 말을 사용할 때는 정확히 무엇에 적용하는지 구분해야 한다.

---

### 6.1 model weight의 EMA

딥러닝에서 성능 향상을 위해 사용하는 EMA는 보통 이 경우다.  
즉, loss가 아니라 **모델 파라미터 자체에 EMA를 적용**한다.

학습 중에는 원본 모델 파라미터를 optimizer로 갱신하고, 동시에 별도의 shadow model을 유지한다.

$$
\theta_t^{EMA} = \beta \theta_{t-1}^{EMA} + (1-\beta)\theta_t
$$
