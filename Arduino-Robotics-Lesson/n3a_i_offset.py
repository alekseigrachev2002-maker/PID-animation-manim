"""Урок 3, сцена 3a. Смещение на дуге и И.

Робот въезжает с верхней прямой в правую внутреннюю петлю (по часовой — поворот направо, R 240–260); старт
за 12 см до дуги — без стартового выброса uP. ПД из урока 2 (kp = 0,2, kd = 1) едет со
смещением: чтобы поворачивать, нужен постоянный control, а его дает только постоянная ошибка — err стоит
полкой около +210, uP ≈ +42. И копит ошибку: uI растет, берет поворот на себя, uP и err уходят к нулю.
"""
from nano_base import NanoRunSeries, render
from nano_track import S_LOOP_RIGHT, ShortTrack
from visuals.panel import Signal

START = S_LOOP_RIGHT - 120.0   # путь по маршруту (по часовой), мм: верхняя прямая, 12 см до правой петли
PD = dict(kp=0.2, kd=1.0)
KI = 0.004


class Nano3a(NanoRunSeries):
    phrase = "Вводим I"
    formula = "control = uP + uD + uI"                              # как в сцене 4a
    formula_colors = [("uP", "e"), ("uD", "D"), ("uI", "I")]
    signals = [Signal("uP = err * kp", "uP", "e"), Signal("uI = integral * ki", "uI", "I")]
    y = ("u", -25, 100)             # uP до ~75 на входе в дугу, на дуге uP или uI ≈ 41
    runs = [
        (dict(PD, ki=0.0), "ki = 0 — uP держит поворот, err ≈ 210"),
        (dict(PD, ki=KI), "ki = 0,004 — uI взял поворот, err → 0"),
    ]

    def make_track(self):
        return ShortTrack()

    def start(self, i):
        return self.track.pose_at(START)


if __name__ == "__main__":
    render(Nano3a, "480p")
