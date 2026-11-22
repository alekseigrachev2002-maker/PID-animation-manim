"""Урок 3, сцена 3a. Смещение на дуге и И.

Робот въезжает с верхней прямой в правую внутреннюю петлю (по часовой — поворот направо, R 240, дальше R 260);
старт — то же место, что в 2b и 2c (S_TURN_START, за 12 см до дуги — без стартового выброса uP). ПД — итог
сцены 2c (kp = 0,06, kd = 1) — едет со смещением: чтобы поворачивать, нужен постоянный control, а его дает только
постоянная ошибка — err уходит к полке ≈ +700 (датчик целиком на линии), uP ≈ +42. И копит ошибку: uI растет до
≈ +57, берет поворот на себя, err уходит к нулю примерно за 0,8 с; на переходе R 240 → R 260 uI оказывается
больше нужного — короткий провал err до −270, и к концу прогона снова err ≈ 0.

ki проверен по модели: 0,003 — err к концу прогона снова −110; 0,004 и 0,005 — err ≈ 0 к концу; 0,006 и 0,008 —
раскачка, err снова +300…+500. Взято 0,005.
"""
from nano_base import NanoRunSeries, render
from n2c_d import KP_ALMOST
from nano_track import S_TURN_START, ShortTrack
from visuals.panel import Signal

START = S_TURN_START
PD = dict(kp=KP_ALMOST, kd=1.0)    # итог 2c — регулятор уроков 3 и 4
KI = 0.005


class Nano3a(NanoRunSeries):
    phrase = "Вводим I"
    formula = "control = uP + uD + uI"                              # как в сцене 4a
    formula_colors = [("uP", "e"), ("uD", "D"), ("uI", "I")]
    signals = [Signal("uP = err * kp", "uP", "e"), Signal("uI = integral * ki", "uI", "I")]
    y = ("u", -25, 100)             # uP до ~42 (полка err ≈ 700), uI до ~57
    runs = [
        (dict(PD, ki=0.0), "ki = 0"),
        (dict(PD, ki=KI), f"ki = {KI:g} — err → 0".replace(".", ",")),
    ]

    def make_track(self):
        return ShortTrack()

    def start(self, i):
        return self.track.pose_at(START)


if __name__ == "__main__":
    render(Nano3a, "480p")
