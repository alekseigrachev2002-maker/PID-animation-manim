"""Все параметры проекта: геометрия, датчики, моторы, регулятор.

Единицы: мм, с, рад. ШИМ — в единицах analogWrite (0…255), АЦП — 0…1023.
Значения с пометкой «заглушка» ждут замеров.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    # начало координат — середина оси колёс, +y вперёд
    chassis_r: float = 80.0
    cutout_x: tuple = (50.0, 74.0)      # вырез под колесо по |x|
    cutout_h: float = 48.0
    cutout_r: float = 4.0
    wheel_d: float = 44.0
    wheel_w: float = 20.0
    wheel_x: float = 62.0               # центр колеса по |x|
    sensor_x: float = 23.789            # оптический центр TCRT по |x|
    sensor_y: float = 94.714            # оптический центр TCRT по y
    mount_y: float = 73.214             # крепёжное отверстие платы
    board_len: float = 40.0
    board_w: float = 10.0
    board_front_to_mount: float = 31.5  # от переднего края платы до крепёжного отверстия

    @property
    def track_width(self) -> float:
        return 2 * self.wheel_x


@dataclass(frozen=True)
class Sensor:
    sigma: float = 2.5          # радиус гауссова пятна, мм
    white: tuple = (120, 120)   # АЦП на белом, (L, R) — заглушка
    black: tuple = (860, 860)   # АЦП на чёрном, (L, R) — заглушка
    noise: float = 0.0          # СКО шума АЦП
    line_width: float = 10.0


@dataclass(frozen=True)
class Motor:
    tau: float = 0.080          # инерция колеса, с — заглушка
    pwm_start: float = 67.0     # трогание стоящего колеса (замер)
    pwm_stop: float = 50.0      # крутящееся колесо останавливается ниже этого (замер)
    v_at_150: float = 300.0     # скорость при ШИМ 150, мм/с — заглушка
    stop_eps: float = 1.0       # |v| ниже этого считаем, что колесо встало, мм/с

    @property
    def k(self) -> float:
        """мм/с на единицу ШИМ выше порога остановки."""
        return self.v_at_150 / (150.0 - self.pwm_stop)


@dataclass(frozen=True)
class Controller:
    speed: float = 150.0
    kp: float = 0.3
    kd: float = 0.0             # на один тик регулятора
    balance: float = 1.0        # sensorBalance
    period: float = 0.005       # период регулятора, с
    pwm_limit: float = 255.0
