---
title: "로봇에서 base, ee 좌표계 변환 매커니즘/구현"
date: 2025-09-26
tags: [Dynamics, robotics]
toc: true
---
# Base, ee 좌표계 변환 매커니즘

## base 좌표계
base 좌표계는 보통 고정된 <i,j,k> 좌표계

## ee 좌표계
end effector 는 trnasversal 한 움직임과 rotation 움직임을 모두 갖는 <i,j,k> 좌표계

그러면 end effector 에서 물체로 가는 경로벡터 $\vec{r}$ 를 어떻게 생성하는가?
transversal 과 rotational 모두를 표현하려면 homogeneous transformation matrix 를 사용한다.

$$
T = 
\begin{bmatrix}
    R & \vec{t} \\
    \vec{0}^T & 1
\end{bmatrix}
\in \mathbb{R}^{4 \times 4}
$$

결국 가장 기본적인 아이디어는 아래의 그림에서 출발한다.
물체가 목표지점, 이동하는 좌표계가 ee, 고정된 좌표계가 base. 여기서 계산한 $\vec{r}$ 가지고 Jacobian 으로 joint angle velocity IK 풀어서 제어하는 것.


{% include figure
   image_path="/assets/images/major/frames.png"
   alt="frames"
   caption="Relative frames"
   class="align-center"
%}