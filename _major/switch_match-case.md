---
title: "switch in C, match - case in python"
date: 2025-09-04
tags: [major study]
toc: true
---
# 프로그램 조건 분기 - switch in C, match - case in Python

C 언어에서 switch 를 사용하면 프로그램을 여러 갈래로 분기시킬 수 있다.

```
#include <stdio.h>

int main() {
    int day = 3; // 1(월요일)부터 7(일요일)까지의 값을 가집니다.

    switch (day) {
        case 1:
            printf("월요일 (Monday)\n");
            break;
        case 2:
            printf("화요일 (Tuesday)\n");
            break;
        case 3:
            printf("수요일 (Wednesday)\n");
            break;
        case 4:
            printf("목요일 (Thursday)\n");
            break;
        case 5:
            printf("금요일 (Friday)\n");
            break;
        case 6:
        case 7: // 6 또는 7일 때 모두 이 코드를 실행합니다.
            printf("주말 (Weekend)\n");
            break;
        default:
            printf("잘못된 입력입니다.\n");
            break;
    }

    return 0;
}

```

이걸 python 3.10 부터 match - case 로 아래와 같이 사용이 가능해졌다.

```
day = "Monday"

match day:
    case "Saturday" | "Sunday": # 여러 패턴을 OR(|)로 묶을 수 있습니다.
        print("It's the weekend.")
    case "Monday":
        print("It's Monday.")
    case "Friday":
        print("Almost the weekend!")
    case _: # C 언어의 'default'에 해당합니다. 일치하는 케이스가 없을 때 실행됩니다.
        print("It's a weekday.")
```
