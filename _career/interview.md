---
title: "현직 interview"
date: 2025-09-16
tags: [Career]
toc: true
---

## LG AI research product 팀 팀장 at LG AI tech connect
어떤 사람을 주로 뽑는지에 대해서, 물론 domain specific 한 knowledge 도 중요하겠지만
한 문제를 깊게 파 본 경험이 있는 사람이 뽑을 때 눈길이 간다. 프로젝트 하면서 문제를 해결 해 본 사람.
이런 경우 교육하고 부서에 배정할 때 어디를 가든 잘 배우고 좋은 퍼포먼스를 내더라.

## 김윤래 멘토님 at 2025 한이음
이번에 IK 를 적용한 로봇 팔을 만들 때 해석적으로는 완벽한 QP 식을 활용해서 IK 를 계산했다. 그런데 데모 찍을 떄는 오류가 안 났고, 3일 뒤에 다시 해 보니 오류가 난다. 거기서 또 3일 뒤에 하니 오류가 나지 않은 경우도 있었다. 이런 경우를 여쭤봤는데

- lowlevel 단에서 로깅을 분석해야 한다. 현업에서는 이 에러를 재현하는 것도 해서 양산에 들어가야 한다.
- one shot problem 이 가장 문제가 된다.
- error 가 날 때의 joint angle 값, current 등을 모두 로그로 남겨서 재현해 보아야 한다.