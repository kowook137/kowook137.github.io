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

\[
\theta_t^{EMA} = \beta \theta_{t-1}^{EMA} + (1-\beta)\theta_t
\]

여기서 각 기호의 의미는 다음과 같다.

- \(\theta_t\): 현재 step \(t\)에서 optimizer에 의해 업데이트된 실제 모델 파라미터
- \(\theta_t^{EMA}\): EMA로 관리되는 smoothed 파라미터
- \(\beta\): decay coefficient 또는 smoothing coefficient  
  - 보통 0.99, 0.999, 0.9999 같은 값이 사용된다

이 식은 현재 파라미터 \(\theta_t\)를 그대로 사용하는 것이 아니라, 직전까지 누적된 EMA 값과 현재 파라미터를 혼합하여 새로운 EMA 파라미터를 만드는 방식이다.

이를 전개하면 다음과 같은 weighted average 형태가 된다.

\[
\theta_t^{EMA}
= (1-\beta)\theta_t + (1-\beta)\beta \theta_{t-1}
+ (1-\beta)\beta^2 \theta_{t-2} + \cdots
\]

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

\[
g_t = \nabla L_{B_t}(\theta_t)
\]

여기서 \(B_t\)는 step \(t\)에서 사용된 mini-batch이다.  
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

질문에서 언급된 “이미 높은 점수인데 EMA 적용 후 더 올랐다”는 결과는 바로 이런 맥락에서 해석할 수 있다.  
즉, 모델의 표현력이 부족해서 성능이 낮았던 것이 아니라, **추정 분산과 parameter fluctuation이 성능 병목이었던 상황**일 수 있다.

---

## 6. EMA가 평균하는 것은 무엇인가

EMA라는 말을 사용할 때는 정확히 무엇에 적용하는지 구분해야 한다.

### 6.1 metric의 EMA

예를 들어 training loss를 부드럽게 보기 위해 다음과 같이 쓸 수 있다.

\[
m_t = \beta m_{t-1} + (1-\beta)x_t
\]

여기서 \(x_t\)는 현재 step의 loss나 reward 같은 scalar 값이다.

이 경우 EMA는 단지 **그래프를 부드럽게 보기 위한 시각화용 smoothing**일 뿐이며, 모델 자체의 성능에는 직접 영향을 주지 않는다.

---

### 6.2 model weight의 EMA

딥러닝에서 성능 향상을 위해 사용하는 EMA는 보통 이 경우다.  
즉, loss가 아니라 **모델 파라미터 자체에 EMA를 적용**한다.

학습 중에는 원본 모델 파라미터를 optimizer로 갱신하고, 동시에 별도의 shadow model을 유지한다.

\[
\theta_t^{EMA} = \beta \theta_{t-1}^{EMA} + (1-\beta)\theta_t
\]

평가할 때는 보통 현재 모델 \(\theta_t\)가 아니라 EMA 모델 \(\theta_t^{EMA}\)를 사용한다.

질문에서 언급된 성능 향상은 거의 확실하게 이 두 번째 경우에 해당한다.

---

## 7. decay coefficient \(\beta\)의 의미

EMA에서 \(\beta\)는 과거 정보를 얼마나 강하게 유지할지를 결정한다.

- \(\beta\)가 작을수록: 최근 파라미터를 빠르게 반영
- \(\beta\)가 클수록: 과거 정보를 오래 유지하며 더 강하게 smoothing

보통 다음과 같은 감각으로 이해할 수 있다.

- \(\beta = 0.9\): 비교적 짧은 구간 평균
- \(\beta = 0.99\): 중간 정도 smoothing
- \(\beta = 0.999\): 긴 구간 smoothing
- \(\beta = 0.9999\): 매우 긴 구간 smoothing

직관적으로는 유효 평균 길이를 대략 다음처럼 볼 수 있다.

\[
\text{effective window} \approx \frac{1}{1-\beta}
\]

예를 들면,

- \(\beta = 0.99\)이면 대략 최근 100 step 수준의 영향
- \(\beta = 0.999\)이면 대략 최근 1000 step 수준의 영향
- \(\beta = 0.9999\)이면 대략 최근 10000 step 수준의 영향

물론 이것은 정확한 sliding window 의미는 아니고, 영향 범위를 직관적으로 이해하기 위한 근사치이다.

---

## 8. EMA의 장점

EMA의 대표적인 장점은 다음과 같다.

### 8.1 구현이 간단하다

기존 optimizer를 바꿀 필요 없이, 별도의 shadow model만 유지하면 된다.  
즉, Adam이나 SGD 같은 optimizer 위에 손쉽게 덧붙일 수 있다.

---

### 8.2 추가 추론 비용이 거의 없다

여러 checkpoint를 ensemble하는 방식은 추론 비용이 커지지만, EMA는 하나의 평균된 모델만 사용하면 된다.  
따라서 성능 향상 대비 비용이 매우 낮은 편이다.

---

### 8.3 validation/test 성능이 더 안정적일 수 있다

학습 후반부에 score가 들쭉날쭉한 경우 EMA 모델은 더 안정된 성능을 보이는 경우가 많다.  
이는 논문 실험에서 결과 재현성과 설득력을 높이는 데도 유리하다.

---

### 8.4 이미 높은 성능 영역에서도 추가 향상을 줄 수 있다

성능이 낮을 때는 모델 capacity나 optimization 자체가 문제일 수 있지만, 높은 성능 구간에서는 작은 variance 감소가 실제 metric 개선으로 이어질 수 있다.  
EMA는 바로 이런 상황에서 의미 있는 차이를 만드는 경우가 있다.

---

## 9. EMA의 한계와 주의점

EMA가 유용하다고 해서 항상 무조건 좋은 것은 아니다.

### 9.1 decay가 너무 크면 학습 초반 적응이 느려질 수 있다

학습 초반에는 파라미터가 빠르게 변화해야 하는데, \(\beta\)가 너무 크면 EMA 모델이 현재 모델을 지나치게 느리게 따라간다.  
따라서 초반에는 EMA가 실제 상태를 잘 반영하지 못할 수 있다.

이 때문에 일부 구현에서는 다음과 같은 전략을 사용한다.

- 일정 step 이후에 EMA 시작
- decay를 warmup 방식으로 점진적으로 증가
- 초반에는 작은 \(\beta\), 후반에는 큰 \(\beta\) 사용

---

### 9.2 non-stationary objective에서는 불리할 수 있다

목표가 시간에 따라 빠르게 변하는 문제에서는, 너무 긴 구간 평균이 오히려 현재 상태를 둔감하게 만들 수 있다.  
즉, EMA가 과거의 오래된 정보를 너무 많이 끌고 가면 최근 변화에 적응이 느려질 수 있다.

---

### 9.3 마지막 모델과 EMA 모델을 혼동하면 안 된다

실험 보고 시에는 다음을 분리해서 비교해야 한다.

- last-step model
- best validation checkpoint
- EMA model

이 셋은 서로 다른 결과를 줄 수 있다.  
EMA가 좋았다고 해서 마지막 모델도 같은 성능일 것이라고 가정하면 안 된다.

---

### 9.4 BatchNorm 등 stateful buffer 처리에 주의가 필요하다

모델에 BatchNorm이 포함되어 있으면 단순히 weight만 EMA할 것인지, running mean/variance 같은 buffer도 함께 처리할 것인지에 따라 결과가 달라질 수 있다.  
따라서 프레임워크 구현이나 검증된 유틸리티를 사용하는 것이 안전하다.

---

## 10. EMA와 Polyak Averaging, SWA의 관계

EMA는 broader한 parameter averaging 계열 기법 중 하나다.

### 10.1 Polyak Averaging

Polyak averaging은 과거 iterate들을 **동일 가중치**로 평균하는 방식이다.  
고전적인 stochastic approximation 이론에서 중요한 방법이다.

---

### 10.2 EMA

EMA는 과거 iterate를 평균하되, **최근 파라미터에 더 큰 가중치**를 부여한다.  
즉, Polyak averaging보다 최근 상태를 더 잘 반영한다.

---

### 10.3 SWA (Stochastic Weight Averaging)

SWA는 학습 후반부의 여러 checkpoint를 특정 전략에 따라 평균하는 기법이다.  
EMA와 목적은 유사하지만, 사용 시점과 평균 방식이 다르다.

정리하면 다음과 같다.

- Polyak averaging: 동일 가중 평균
- EMA: 지수 감소 가중 평균
- SWA: 특정 시점의 checkpoint들을 선택적으로 평균

EMA는 이들 중에서도 구현이 가장 간단하고, 일반적인 training loop에 쉽게 삽입할 수 있다는 장점이 있다.

---

## 11. 실제 학습 과정에서 EMA는 어떻게 동작하는가

학습 과정은 보통 다음처럼 진행된다.

1. 원본 모델 파라미터 \(\theta\)를 optimizer로 업데이트
2. 업데이트된 \(\theta\)를 이용해 EMA 파라미터 \(\theta^{EMA}\) 갱신
3. 다음 step 반복

즉, 학습은 어디까지나 원본 모델이 수행하고, EMA는 그 결과를 별도의 shadow copy에 축적한다.

이 구조를 식으로 쓰면 다음과 같다.

### 11.1 원본 모델 업데이트

\[
\theta_t = \theta_{t-1} - \eta g_t
\]

여기서 \(\eta\)는 learning rate, \(g_t\)는 현재 batch에서 계산된 gradient이다.

### 11.2 EMA 모델 업데이트

\[
\theta_t^{EMA} = \beta \theta_{t-1}^{EMA} + (1-\beta)\theta_t
\]

즉, optimizer는 원본 모델에만 적용되고, EMA 모델은 gradient를 직접 사용하지 않는다.  
단지 원본 모델의 시간적 평균을 추적하는 역할만 한다.

---

## 12. 왜 "이미 높은 점수인데 더 오른다"는 것이 가능한가

겉으로 보면 이미 높은 성능에 도달한 모델이라면 EMA로 더 오르기 어려워 보일 수 있다.  
하지만 실제로는 매우 자연스러운 현상이다.

그 이유는 다음과 같다.

### 12.1 높은 점수라고 해서 variance가 사라진 것은 아니다

모델이 좋은 해 근처에 도달했더라도, mini-batch noise로 인해 그 주변에서 계속 흔들릴 수 있다.  
즉, mean performance는 높지만 variance가 남아 있는 상태일 수 있다.

EMA는 이 variance를 줄여준다.  
따라서 이미 높은 성능 영역에서도 추가 향상이 가능하다.

---

### 12.2 조합 최적화 문제는 작은 파라미터 차이가 큰 score 차이를 만들 수 있다

3D bin packing에서는 placement sequence가 조금만 달라져도 utilization이나 packing success가 변할 수 있다.  
따라서 "미세한 smoothing"이 곧 "실질적인 성능 차이"로 이어질 수 있다.

---

### 12.3 마지막 모델은 최고 성능 지점이 아닐 수 있다

학습 trajectory 상에서 좋은 상태가 여러 번 나타났더라도, 마지막 step이 반드시 가장 좋은 지점이라는 보장은 없다.  
EMA는 그 좋은 상태들을 일부 평균에 반영하므로, 결과적으로 마지막 checkpoint보다 더 나은 모델이 될 수 있다.

---

## 13. 논문에서 EMA를 설명할 때 사용할 수 있는 기술적 해석

논문 문장으로 정리하면 EMA는 다음과 같이 설명할 수 있다.

> Exponential Moving Average (EMA) maintains a temporally smoothed copy of model parameters by exponentially averaging the current parameters over training steps. This reduces the variance induced by stochastic mini-batch updates and often yields a more stable and better-generalizing model than the last-step weights.

이를 한국어로 기술하면 다음과 같다.

> EMA는 학습 step에 따라 변화하는 모델 파라미터의 지수이동평균을 유지하는 기법으로, stochastic mini-batch update가 유발하는 파라미터 변동성을 줄이고 마지막 step의 파라미터보다 더 안정적이며 일반화가 잘 되는 모델을 제공할 수 있다.

3D bin packing과 같은 맥락에 맞춰 더 구체적으로 쓰면 다음과 같이 표현할 수 있다.

> In our mini-batch training setting, the model parameters exhibit non-negligible fluctuations due to stochastic gradient noise. EMA mitigates such fluctuations by maintaining a smoothed parameter trajectory, which leads to more stable decision behavior and improved packing performance.

한국어로는 다음과 같다.

> 본 연구의 mini-batch 학습 환경에서는 stochastic gradient noise로 인해 모델 파라미터가 유의미하게 흔들린다. EMA는 이러한 파라미터 trajectory를 시간적으로 smoothing함으로써 의사결정의 안정성을 높이고, 결과적으로 packing 성능을 향상시킨다.

---

## 14. 본 실험 결과에 대한 해석

본 실험에서 EMA를 적용한 모델이 EMA를 사용하지 않은 모델보다 월등히 높은 성능을 보였다는 것은 다음과 같이 해석할 수 있다.

첫째, 모델의 기본적인 표현력은 이미 충분했지만, mini-batch 학습으로 인해 파라미터 variance가 남아 있었을 가능성이 높다.

둘째, 3D bin packing 문제는 조합적 구조와 순차적 의사결정 특성 때문에 파라미터의 작은 흔들림이 최종 packing quality에 직접 영향을 주는 문제일 수 있다.

셋째, EMA는 마지막 step의 모델이 아니라 학습 trajectory 전체에서 비교적 안정적인 영역을 대표하는 파라미터를 형성했을 가능성이 있다.

따라서 EMA에 의한 성능 향상은 단순한 우연한 실험 결과라기보다, 다음과 같은 결론으로 정리할 수 있다.

- mini-batch update의 stochasticity가 실제 성능 병목이었다
- EMA가 이를 완화하여 더 robust한 모델을 형성했다
- 특히 고성능 구간에서 남아 있던 variance를 줄여 추가 score 향상을 달성했다

---

## 15. 핵심 요약

EMA는 딥러닝에서 모델 파라미터를 지수이동평균하여 별도의 smoothed model을 유지하는 기법이다.  
이는 mini-batch 학습에서 발생하는 stochastic update noise를 줄여, 마지막 step의 파라미터보다 더 안정적이고 일반화가 잘 되는 모델을 제공할 수 있다.

특히 3D bin packing과 같은 조합 최적화 문제에서는 작은 파라미터 변화가 최종 score에 큰 차이를 만들 수 있기 때문에, EMA의 효과가 더욱 크게 나타날 수 있다.  
따라서 본 실험에서 EMA 적용 모델이 비적용 모델보다 월등히 높은 성능을 보인 것은, EMA가 파라미터 trajectory를 효과적으로 smoothing하여 decision stability와 generalization을 동시에 개선했기 때문으로 해석할 수 있다.

---

## 16. 한 줄 정리

EMA는 **현재 모델의 noisy한 파라미터를 직접 쓰지 않고, 그 파라미터의 시간적 지수평균을 사용하여 더 안정적이고 성능이 좋은 평가용 모델을 만드는 기법**이다.