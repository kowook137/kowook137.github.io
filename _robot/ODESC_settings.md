---
title: "ODESC 및 BLDC 모터 세팅팅"
date: 2025-09-05
tags: [Robot, Humanoid, frimware, BLDC, ODESC]
toc: true
---
# ODESC 와  BLDC 모터


{% include figure
   image_path="/assets/images/robot/ODESC.png"
   alt="ODESC board"
   caption="Tickle bot의 ODESC board"
   class="align-center"
%}

ODESC board 는 기본적으로 odrivetool 이라는 툴을 활용해서 설정한다. 보통 ODESC 보드를 받으면 firmware 는 설치되어 받으니, 어떤 버전이 깔려있는지 확인하고, firmware 버전과 compatible 한 odrivetool 버전을 설치해야 설정할 수 있다.

아래의 링크에서 compatibility 를 체크하자.
```
https://github.com/odriverobotics/ODrive/releases
```

odirve 를 설치했다면, BLDC 모터를 ODESC 보드에 연결해야 한다.

tickle bot 에 사용된 ODESC board 는
"ODESC 3.6 56V"
전압에 맞게 승압/강압을 해 주어야 한다.
axis1, axis0 에 맞게 전원과 hall sensor 를 연결해야 한다.

## BLDC 모터 선 구성
BLDC 모터에서 굵은 선 3개, 얇은 선 3개가 나온다. 
```
굵은선: 전원선
얇은선: Hall sensor 선
```
Hall sensor 선의 경우 어차피 calibration 과정에서 상대적으로 매핑되기 때문에 순서가 약간 바뀌어도 상관없다.
또한, 양수로 속도를 준 상태에서 회전 방향을 반대로 바꾸고 싶다면 direction 부호를 설정에서 바꾸어도 되지만 Hall sensor 에서 선 매핑을 ABZ 에서 순서를 약간 바꾸어도 되는 것 같다.

calibration 은 아래와 같이 진행한다.

```
odrv0.erase_configuration() # with reboot
# lease the protection
odrv0.config.dc_max_negative_current = -5.0 # charging current
odrv0.axis0.motor.config.current_lim = 20

# set motor poles and hall encoder
odrv0.axis0.motor.config.pole_pairs = 10 # the motor has 10 poles
odrv0.axis0.encoder.config.mode = ENCODER_MODE_HALL # use ABC(UVW) hall encoder
odrv0.axis0.encoder.config.cpr = 60 # pulses per rev (6 * pole num)
odrv0.axis0.encoder.config.bandwidth = 50

odrv0.axis0.motor.config.resistance_calib_max_voltage = 5 # voltage for calibration
odrv0.axis0.encoder.config.calib_range = 0.1
odrv0.axis0.encoder.config.calib_scan_distance = 150
odrv0.axis0.motor.config.current_control_bandwidth = 1000 # related to the PD parameter of the current loop. the larger it is, the "harder" the current loop is.

# ignore illegal_hall_state error
odrv0.axis0.encoder.config.ignore_illegal_hall_state = True

odrv0.axis1.motor.config.current_lim = 20

odrv0.axis1.motor.config.pole_pairs = 10
odrv0.axis1.encoder.config.mode = ENCODER_MODE_HALL
odrv0.axis1.encoder.config.cpr = 60
odrv0.axis1.encoder.config.bandwidth = 50

odrv0.axis1.motor.config.resistance_calib_max_voltage = 5 
odrv0.axis1.encoder.config.calib_range = 0.1
odrv0.axis1.encoder.config.calib_scan_distance = 150
odrv0.axis1.motor.config.current_control_bandwidth = 1000

odrv0.axis1.encoder.config.ignore_illegal_hall_state = True

# wait for 1 second

odrv0.save_configuration()
odrv0.reboot()

odrv0.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE # start calibration

odrv0.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE # 开始标定

# wait until motor stop

odrv0.axis1.encoder.config.pre_calibrated = True
odrv0.axis1.motor.config.pre_calibrated = True

# wait for 1 second

odrv0.save_configuration()
odrv0.reboot()

# position mode and tune the parameters
odrv0.axis0.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
odrv0.axis0.controller.config.pos_gain = 20
odrv0.axis0.controller.config.vel_gain = 0.10
odrv0.axis0.controller.config.vel_integrator_gain = 0

# trapezoidal acceleration and deceleration:
odrv0.axis0.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
odrv0.axis0.controller.config.vel_limit = 20
odrv0.axis0.trap_traj.config.vel_limit = 20
odrv0.axis0.trap_traj.config.accel_limit = 1
odrv0.axis0.trap_traj.config.decel_limit = 1

# # start closed loop control
# odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# odrv0.axis0.config.startup_closed_loop_control = True

# # test
# odrv0.axis0.controller.input_pos = 5

# position mode and tune the parameters
odrv0.axis1.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
odrv0.axis1.controller.config.pos_gain = 20
odrv0.axis1.controller.config.vel_gain = 0.10
odrv0.axis1.controller.config.vel_integrator_gain = 0

# trapezoidal acceleration and deceleration:
odrv0.axis1.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
odrv0.axis1.controller.config.vel_limit = 20
odrv0.axis1.trap_traj.config.vel_limit = 20
odrv0.axis1.trap_traj.config.accel_limit = 1
odrv0.axis1.trap_traj.config.decel_limit = 1

# # start closed loop control
# odrv0.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# odrv0.axis1.config.startup_closed_loop_control = True

# # test
# odrv0.axis1.controller.input_pos = 0

# wait for 1 second

odrv0.save_configuration()
odrv0.reboot()

```

위와 같이 calibration 을 진행하면 되고,
pole pair, cpr 값 등은 사용하는 BLDC 모터에 맞추면 된다.

Ticklebot BLDC 모터의 경우 위의 설정만으로는 부드럽게 돌아가지 않았다. 양 쪽이 동일한 속도 명령에도 도는 회전수가 약간 다르거나 부드럽지 않게 도는 경우가 발생했다. 아래의 설정을 추가하면 된다.

```
# (필요시) 전류루프/컨트롤 파라미터 정렬
odrv0.axis1.motor.config.current_control_bandwidth = 300
odrv0.axis1.controller.config.vel_gain = 0.03
odrv0.axis1.controller.config.vel_integrator_gain = 0.05

# 램프 세팅
odrv0.axis1.controller.config.control_mode   = CONTROL_MODE_VELOCITY_CONTROL
odrv0.axis1.controller.config.input_mode     = INPUT_MODE_VEL_RAMP
odrv0.axis1.controller.config.vel_ramp_rate  = 1.5
odrv0.axis1.controller.config.vel_limit      = 6.0
odrv0.save_configuration()

# 테스트
odrv0.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
odrv0.axis1.controller.input_vel = 2.0
import time; time.sleep(2.0)
odrv0.axis1.controller.input_vel = 0.0
time.sleep(1.0)
dump_errors(odrv0)
odrv0.axis1.requested_state = AXIS_STATE_IDLE
```

속도 램프를 soft ramp 로 설정하면 덜컹거림이 완화되었던 것 같다.

