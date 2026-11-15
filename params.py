"""Все параметры проекта: геометрия, датчики, моторы, регулятор.

Единицы: мм, с, рад. ШИМ — в единицах analogWrite (0…255), АЦП — 0…1023.
Откуда каждое значение — в комментарии: замер, оценка или подобрано. Сводка замеров — docs/measurements.md.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    # начало координат — середина оси колес, +y вперед
    chassis_r: float = 80.0
    cutout_x: tuple = (50.0, 74.0)      # вырез под колесо по |x|
    cutout_h: float = 48.0
    cutout_r: float = 4.0
    wheel_d: float = 43.0               # как WHEEL_D в тестах машинки
    wheel_w: float = 20.0
    wheel_x: float = 62.0               # центр колеса по |x|
    sensor_x: float = 23.789            # оптический центр TCRT по |x|
    sensor_y: float = 94.714            # оптический центр TCRT по y
    mount_y: float = 73.214             # крепежное отверстие платы
    board_len: float = 40.0
    board_w: float = 10.0
    board_front_to_mount: float = 31.5  # от переднего края платы до крепежного отверстия
    axle_w: float = 10.0                # видимая ось: толщина, в зазоре между акрилом и колесом

    @property
    def track_width(self) -> float:
        return 2 * self.wheel_x


@dataclass(frozen=True)
class Sensor:
    sigma: float = 2.5          # радиус гауссова пятна, мм
    white: tuple = (939, 939)   # АЦП на белом, (левый, правый): среднее замера 920 и 958; белое выше
    black: tuple = (204, 204)   # АЦП на черном, (левый, правый): среднее замера 191 и 216
    swapped: bool = False       # True — A0 (L в коде) на физически правом датчике; проверено: A0 — левый
    noise: float = 0.0          # СКО шума АЦП
    line_width: float = 30.0   # поле 2400×1500 (PDF): линия 30 мм


@dataclass(frozen=True)
class Motor:
    tau: float = 0.085          # инерция колеса, с: подобрана вместе с delay под выбег 30 мм с 280 мм/с (замер)
    pwm_start: float = 67.0     # трогание стоящего колеса (ранний замер)
    # скорость крутящегося колеса на полу от ШИМ, (ШИМ, мм/с), между точками — прямая: 150 → 280, 255 → 435
    # (замер driveTest); ниже ШИМ 50 крутящееся колесо останавливается (ранний замер)
    curve: tuple = ((50.0, 0.0), (150.0, 280.0), (255.0, 435.0))
    # доля разницы скоростей колес, которая доходит до поворота: остальное съедают шарик и колеса, скребущие пол.
    # Оценка по замеру разворота motors(150, -150) 3 с: колеса прошли ~465 мм (без трения ~840); замер с кабелем —
    # переснять без него (docs/measurements.md, «Трение в повороте»)
    turn_share: float = 0.55
    stop_eps: float = 1.0       # |v| ниже этого считаем, что колесо встало, мм/с
    delay: float = 0.020        # задержка команда → колесо, с: оценка, не замер; одна для всех сцен (≈ 2 периода регулятора)
    swapped: bool = False       # motors(левый, правый): первый аргумент — левое колесо (M2), проверено на машинке

    def speed(self, pwm) -> float:
        """Установившаяся скорость крутящегося колеса при ШИМ pwm, мм/с (со знаком)."""
        pw, v = zip(*self.curve)
        return float(np.sign(pwm) * np.interp(abs(pwm), pw, v, left=0.0))


@dataclass(frozen=True)
class Controller:
    speed: float = 150.0
    kp: float = 0.09            # найдены по модели (tune.py), как в сценах 2c и 3
    kd: float = 1.0             # на один тик регулятора
    balance: float = 1.0        # sensorBalance
    period: float = 0.0103      # период регулятора, с: delay(10) + два analogRead (~0,11 мс каждый) + расчет и motors() (~0,05 мс)
    pwm_limit: float = 255.0
    threshold: float = 300.0    # порог перекрестка (урок 3, #define THRESHOLD 300): L и R оба ниже — перекресток


@dataclass(frozen=True)
class Style:
    # шрифты и кегли — как на слайдах презентации (видео идет на весь слайд 16:9, 1 pt слайда = 4/3 px кадра 1280×720)
    font: tuple = ("Segoe UI", "Open Sans", "Inter")                                     # первый найденный
    code_font: tuple = ("Consolas", "JetBrains Mono", "DejaVu Sans Mono", "Courier New")  # первый найденный
    title_px: float = 43.0          # заголовок — 32 pt, полужирный, как заголовки слайдов
    title_weight: str = "BOLD"
    text_px: float = 24.0           # текст и формулы — 18 pt
    small_px: float = 19.0          # мелкие подписи не по сути (числа на осях, единицы, порог) — 14 pt
    L: str = "#4C8DF6"
    R: str = "#F59E0B"
    e: str = "#34D399"
    D: str = "#E879F9"
    graph_bg: str = "#0E1117"
    field: str = "#EEECE5"
    line: str = "#151515"
    mark: str = "#FFDF87"           # желтая разметка поля (квадраты)
    mark_mm: float = 5.0            # толщина разметки, мм
    text: str = "#D7DCE5"
    muted: str = "#8A94A6"
    axis: str = "#3A3F48"
    block: str = "#1A1F29"
    chassis: str = "#32363D"        # шасси, матовое
    chassis_edge: str = "#7A828F"   # кант по краю шасси
    glow: float = 0.75              # яркость ореола метки датчика при сигнале «черный»
    wheel: str = "#1D1D1F"
    tread: str = "#6E6E73"


@dataclass(frozen=True)
class Layout:
    """Макет в пикселях эталонного кадра 1280×720 (начало — левый верхний угол)."""
    width_px: float = 1280.0
    px_per_mm: float = 1.6                    # масштаб робота
    graph_w_px: float = 853.0                 # левая панель графиков, 2/3 ширины
    robot_px: tuple = (1066.5, 440.0)         # середина оси колес на экране (ниже центра — больше видно впереди)
    plot_px: tuple = (96.0, 168.0, 780.0, 560.0)  # область графика: x0, y0, x1, y1
    header_y_px: float = 50.0                 # середина строки заголовка (и блока формулы)
    caption_y_px: float = 104.0               # середина строки пояснения
    legend_y_px: float = 622.0                # середина строки легенды
    readout_y_px: float = 670.0               # середина строки ШИМ колес
    margin_px: float = 24.0
