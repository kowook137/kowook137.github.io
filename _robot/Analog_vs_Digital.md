---
title: "DOGlove.md"
date: 2026-03-16
tags: [robot, teleoperation]
toc: true
---

# Analog Signal vs Digital Signal (Sensor Wiring)

## 1. 왜 아날로그 선은 길게 뽑으면 안 되는가

아날로그 신호는 **전압 자체가 값(value)** dlek.

예시

- 1.2 V → 특정 각도
- 2.0 V → 다른 각도

문제는 **노이즈가 전압에 직접 영향** 을 주기 때문에, 센서가 노이즈에 민감하다.

예

실제 신호 : 1.20 V  
노이즈   : ±0.05 V  
측정값   : 1.15 ~ 1.25 V  

즉 다음 요소들이 신호 왜곡을 유발한다.

- Electromagnetic Interference (EMI)
- 케이블 저항
- 커패시턴스
- 접지 문제

따라서 일반적으로 아날로그 선은 수십 cm 이내로 짧게 유지한다.

---

# 2. 디지털 신호는 왜 길게 뽑아도 되는가

디지털 신호는 **전압 자체가 값이 아니라 상태(state)** 이다.

예

0 V   → LOW  
3.3 V → HIGH  

노이즈가 발생해도 상태 구분이 유지된다. 이산적인 상태이기 때문에.

예

3.3 V → 3.1 V (여전히 HIGH)  
0 V   → 0.2 V (여전히 LOW)  

따라서 디지털 신호의 특징

- 노이즈에 강함
- 장거리 전송 가능
- 에러 검출 가능

일반적인 통신 방식

| Interface | 특징 |
|---|---|
| UART | 수 m 가능 |
| I2C | 보통 1 m 이하 |
| SPI | 보통 짧은 거리 |
| RS485 | 수십 m 가능 |
| CAN | 수십 ~ 수백 m |

---

# 3. 어떤 센서가 아날로그 / 디지털인지 구분하는 방법

## (1) 데이터시트 확인

### 아날로그 센서

데이터시트에 다음과 같은 표현이 있음

Output: Analog Voltage  
Output range: 0–3.3 V  
Output: 4–20 mA  

대표 예

| 센서 | 출력 |
|---|---|
| Potentiometer | Voltage |
| Strain Gauge | Voltage |
| Analog Pressure Sensor | Voltage |
| Analog Force Sensor | Voltage |

---

### 디지털 센서

데이터시트에 다음 인터페이스가 명시됨

Interface: I2C  
Interface: SPI  
Interface: UART  
Interface: CAN  
Interface: PWM  

대표 예

| 센서 | 인터페이스 |
|---|---|
| AS5600 | I2C |
| MA730 | SPI |
| AMT21 | UART |
| Dynamixel | TTL / RS485 |

---

# 4. 핀 구성으로 구분하는 방법

## 아날로그 센서

보통 3핀

VCC  
GND  
OUT  

OUT → 아날로그 전압

---

## 디지털 센서

### I2C

VCC  
GND  
SDA  
SCL  

### SPI

VCC  
GND  
MISO  
MOSI  
SCLK  
CS  

### UART

VCC  
GND  
TX  
RX  

---

# 5. 코드로 구분하는 방법

MCU 코드에서도 확인 가능

### 아날로그 센서

다음 함수 사용

analogRead()  
ADC_Read()  
HAL_ADC_Start()  

→ ADC 사용

---

### 디지털 센서

다음 인터페이스 사용

I2C  
SPI  
UART  
CAN  

예

HAL_I2C_Master_Transmit()  
SPI_Read()  
UART_Read()  

---

# 6. 로봇 시스템 설계에서의 실제 구조

아날로그 센서는 보통 **센서 근처에서 디지털화**한다.

일반적인 구조

Encoder  
↓ (analog, 짧은 거리)  
Local MCU  
↓ (digital communication)  
Main Controller  

디지털 통신

CAN  
RS485  
UART  

---

# 7. 로봇에서 자주 사용하는 디지털 Encoder

| Encoder | Interface |
|---|---|
| MA730 | SPI |
| AS5048 | SPI |
| AMT21 | UART |
| AS5600 | I2C |

장점

- 노이즈 강함
- 높은 해상도
- 긴 케이블 사용 가능

---

## Source
- DOGlove 제작하면서 배운 것들.

{% include figure
   image_path="/assets/images/robot/DOGlove.png"
   alt="DOGlove"
   caption="DOGlove"
   class="DOGlove"
%}