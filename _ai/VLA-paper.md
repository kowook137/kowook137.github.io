---
title: "VLA 개선을 위해 참조해야 할 논문들"
date: 2025-09-16
tags: [AI, Robot, Physical AI]
toc: true
---

## 방향성
 우선 임베디드 단에서 동작할 수 있는 경량화를 하면서 성능을 유지/개선 할 수 있어야 할 것 같다. 아래의 논문들은 이를 위해 찾은 논문들.

## Qwen3 next
 기본적으로 LLM 모델의 성과이다. 3B 의 파라미터로 256B 파라미터 모델과 유사한 성능을 낼 수 있는 아키텍처를 제안했다. 이 아키텍처를 활용하면 VLA 모델의 경량화를 하면서 성능을 SOTA 모델과 동등하게 유지할 수 있지 않을까?
 expert 를 더 자주 배치하는 아키텍처인 듯 하다. paper 는 찾아봤는데 안 나온다.

```
https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list
```

## Machine Unlearning
 machine unlearning 이라는, "어떻게 모델이 효과적으로 잊게 할 것인가" 를 다룬 논문이다. 코드 또한 아래 링크에서 볼 수 있다. unlearning 기법을 적절히 적용하면 경량화된 모델을 구축하면서 성능에 크게 손실이 일어나지 않을 수 있지 않을까?

 ```
 https://github.com/naver-ai/negmerge
 https://arxiv.org/abs/2410.05583
 ```

 ## CoT-VLA
 기존에 VLA 에서 CoT 를 텍스트로 하던 것과 달리, vision 데이터를 활용해서 CoT 를 하는 것. 이 아키텍처를 활용하면 기존에 액션이 없는 비전 데이터를 활용할 수 있어서 학습할 수 있는 데이터의 양이 획기적으로 늘어난다.

```
https://arxiv.org/abs/2503.22020
```

