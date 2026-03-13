---
title: "Dynamixel 사용"
date: 2026-03-13
tags: [Dynamixel, Robotics]]
toc: true
---

## Dynamixel

대부분의 로봇이 dynamixel 을 사용한다. 원래 feetech 3215 만 쓸 줄 알았는데 dynamixel을 쓸 줄 아는 것이 훨씬 확장성이 좋은 것 같다.

### 1. Motor ID 부여

Dynamixel 의 기본적인 설정은 Dynamixel Wizard 로 이루어진다. 
여기서 각 위치의 모터마다 코드를 보고, 코드에 적합하게 할당된 Motor ID 를 부여하면 된다.

> https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/


{% include figure
   image_path="/assets/images/robot/Dynamixel.png"
   alt="Dynamixel wizard"
   caption="Dynamixel Wizard"
   class="Dynamixel wizard"
%}