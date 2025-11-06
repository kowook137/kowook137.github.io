---
title: "Sergey Levin"
date: 2025-11-06
tags: [robot, Hardware, Bambulab A1]
toc: true
---
# Bambu Lab A1 mini 사용

## 3D printer

 Bambu lab A1 mini 를 활용하여 문서를 출력하는 방법에 대한 설명.

## Step by Step
### 파일 준비
보통 3D 파일은 .STEP, .STL 등으로 모델링 툴에서 나온다. Bambu lab A1 은 적층형이기 때문에 .gcode 의 형태로 slice 한 형태로 출력해야 한다.

> 1) STL 파일 준비

Lerobot 출력이 목적이기 때문에 아래 링크에서 STL 파일을 랩탑에 다운 받는다. 한꺼번에 묶인 파일들은 A1 mini 에 사이즈가 맞지 않으므로 각 파츠 파일들이 따로 있는 아래 링크에서 다운받는다.
```
https://github.com/TheRobotStudio/SO-ARM100?tab=readme-ov-file#printing-the-parts
```

여기 링크에 보면 README 중간에 This table contains all individual files 토글로 각 파일들이 따로 있다. 여기서 출력이 필요한 파일들을 다운 받는다.

> 2) Bambu Lab 프로그램 다운로드

https://bambulab.com/en/download
이 링크에서 Bambu lab studio 를 랩탑에 설치한다.


> 3) 파일 import, 서포트 생성, slice

파일 import: win 파일 탐색기에서 studio 로 드래그 하면 된다.


{% include figure
   image_path="/assets/images/robot/Bambu studio.png"
   alt="Bambu studio"
   caption="Bambu studio software"
   class="align-center"
%}

위와 같은 화면에서 프린트 하려는 파츠들이 나타난다. 

서포트 생성: 보통 대부분의 형상의 경우에, 서포트가 필요하다. 화면 좌측에 보이는 서포트 생성에서 서포트 활성화 -> 트리(자동)을 선택한다.

플레이트 슬라이스: 우측 위에 초록색 버튼으로 플레이트 슬라이스를 한다. 이 단계에서 문제 없이 파일을 출력할 수 있는지 점검할 수 있다.

플레이트 내보내기: 플레이트 슬라이스 버튼 옆에 아랫방향 화살표를 누르면 해당 버튼 위치에 여러 기능이 있다. 여기서 슬라이스 파일 내보내기를 선택해서 Bambu lab A1 mini 에 꽂을 micro sd card 에 저장한다.

> 4. 기타 사항

layer 를 적층해서 출력하는 방식이기 때문에 힘을 받는 방향에 나란하게 layer 가 형성되도록 해야 안정적이다. 만약 layer 와 stress 방향이 수직인 경우 쪼개짐이 발생할 수 있다. 파일 회전, 바닥에 붙이기 등은 Bambulab studio 내부에서 지원한다.

SD 카드는 프린터 전원을 끈 상태에서, 우측 슬롯에서 뽑으면 된다. 전원 버튼은 뒤쪽 좌측에 있다.