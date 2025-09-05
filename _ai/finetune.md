---
title: "Exaone-1.2B finetuning"
date: 2025-09-05
tags: [AI, LLM]
toc: true
---

VLA 나 LBM 등 diffusion, attention mechanism 등에 대한 본질적인 이해가 필요할 것 같아서 수학과 LLM 연구실에 들어왔다.

exaone-1.2B 를 불경 데이터 셋을 기반으로 one-column 으로 finetuning 진행했다.
finetuning 기법은 LoRA.
Chat interaction 은 streamlit 으로 했다.

RAG 기반 응답을 잘 하고, 실행 속도도 빠른 듯 하다.

```
https://huggingface.co/kowook-137/exaone-4.0-1.2B-buddha-merged-bf16
```

{% include figure
   image_path="/assets/images/ai/finetune.png"
   alt="finetune"
   caption="finetuning of Exaone-1.2B using LoRA"
   class="align-center"
%}