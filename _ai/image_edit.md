---
title: "Image CoT, Image edit"
date: 2025-09-09
tags: [AI, AGI, multi-modality]
toc: true
---
아래의 내용은 sudoremove 채널 영상의 요약.
https://www.youtube.com/watch?v=JCYBTGBXQew

## reporting bias
LLM 이 학습하는 텍스트 데이터들은 보통 일반적이지 않은 사건들만을 담고 있다. 예컨대 친구 집에 갔다 온 경우 "소파가 있었다.", "TV가 있었다." 등의 진술은 담기지 않는다.

이것을 reporting bias 라 한다.

때문에 text 만을 학습한 LLM은 성능의 한계가 있다.

## Image CoT
diffusion 기반의 모델들은 image 편집을 잘 못한다. instruction을 잘 따라가지 못 한다.
특정 부분만 바꿔달라고 하면 다른 부분까지 다 바꿔버리는 경우 발생.

Image-edit 모델, 최근 nanno bannana 같은 모델들은 이미지의 작은 부분을 바꿀 수 있다. 때문에 Image 공간상에서의 CoT 가 가능하다.

예컨대 미로 문제를 푸는 경우를 보자. 기존 diffusion 기반의 모델들은 미로 문제를 잘 못 풀 수 있다. 벽의 구조 등을 바꿔버릴 수 있기 때문에. 하지만 image edit 모델의 경우 이러한 문제가 없고, 시각 공간상에서의 CoT 가 가능할 수 있다. 이는 reporting bias 로 인해 발생하는 정보량 input 의 병목을 해결할 수 있다.

즉, consistency 를 유지하면서 vision 공간 안에서 생각/추론 하는 것이 현재의 난제.

## Xiao Jun Podcast 요약
LLM 모델이 매우 커지면, 자꾸 추론 과정을 생략해서 오히려 수학적 추론 능력이 떨어질 수 있더라. 소위 말해서 "건방져(?)" 지는 경향.

nanno bannana 는 autoregressive 로 예상이 된다.

### Xiao jun podcast
https://youtu.be/vWrYHvSRz0s?si=Djz80Cr5ED5R0C2B

## Qwen image edit 모델
https://huggingface.co/Qwen/Qwen-Image-Edit

