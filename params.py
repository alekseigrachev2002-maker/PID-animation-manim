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
    white: tuple = (860, 860)   # АЦП на белом, (L, R) — заглушка; белое выше (схема: фототранзистор вверху, pull-down)
    black: tuple = (120, 120)   # АЦП на чёрном, (L, R) — заглушка
    noise: float = 0.0          # СКО шума АЦП
    line_width: float = 10.0


@dataclass(frozen=True)
class Motor:
    tau: float = 0.080          # инерция колеса, с — заглушка
    pwm_start: float = 67.0     # трогание стоящего колеса (замер)
    pwm_stop: float = 50.0      # крутящееся колесо останавливается ниже этого (замер)
    v_at_150: float = 300.0     # скорость при ШИМ 150, мм/с — заглушка
    stop_eps: float = 1.0       # |v| ниже этого считаем, что колесо встало, мм/с
    delay: float = 0.0          # чистая задержка команда → колесо, с (кратна периоду регулятора)
    swapped: bool = True        # motors(speed1, speed2): speed1 (IN1/IN2) крутит ПРАВОЕ колесо — заглушка до теста 0

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


@dataclass(frozen=True)
class Style:
    font: str = "Inter"
    L: str = "#4C8DF6"
    R: str = "#F59E0B"
    e: str = "#34D399"
    D: str = "#E879F9"
    graph_bg: str = "#0E1117"
    field: str = "#EEECE5"
    line: str = "#151515"
    text: str = "#D7DCE5"
    muted: str = "#8A94A6"
    axis: str = "#3A3F48"
    block: str = "#1A1F29"
    chassis: str = "#262A31"        # глянцевый чёрный: основной тон
    chassis_hi: str = "#555C68"     # глянцевый чёрный: светлая сторона градиента
    chassis_edge: str = "#7A828F"   # кант по краю шасси
    glare: str = "#FFFFFF"          # блик
    glare_opacity: float = 0.30
    chassis_finish: str = "flat"    # "gloss" — градиент и блик, "flat" — ровный матовый
    chassis_flat: str = "#32363D"   # цвет матового шасси
    xray: bool = False              # линия под шасси пунктиром поверх корпуса
    xray_color: str = "#9AA1AD"
    xray_opacity: float = 0.55
    glow: float = 0.75              # яркость ореола метки датчика при сигнале «чёрный»
    chassis_opacity: float = 1.0
    wheel: str = "#1D1D1F"
    tread: str = "#6E6E73"


@dataclass(frozen=True)
class Layout:
    """Макет в пикселях эталонного кадра 1280×720 (начало — левый верхний угол)."""
    width_px: float = 1280.0
    px_per_mm: float = 1.6                    # масштаб робота (было 2.2)
    graph_w_px: float = 853.0                 # левая панель графиков, 2/3 ширины
    robot_px: tuple = (1066.5, 440.0)         # середина оси колёс на экране (ниже центра — больше видно впереди)
    plot_px: tuple = (84.0, 132.0, 809.0, 602.0)  # область графика: x0, y0, x1, y1
    header_y_px: float = 46.0
    legend_y_px: float = 688.0
    margin_px: float = 24.0
