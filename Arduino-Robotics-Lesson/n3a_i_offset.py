"""Урок 3, сцена 3a. Смещение на дуге и И.

Правая внутренняя петля (по часовой — поворот направо, R 240–260). ПД из урока 2 (kp = 0,2, kd = 1) едет со
смещением: чтобы поворачивать, нужен постоянный control, а его дает только постоянная ошибка — err стоит
полкой около +210, uP ≈ +42. И копит ошибку: uI растет, берет поворот на себя, uP и err уходят к нулю.
"""
from nano_base import NanoRunSeries, render
from nano_track import ShortTrack
from visuals.panel import Signal

START = 3420.0        # путь по маршруту (по часовой), мм: начало правой внутренней петли
PD = dict(kp=0.2, kd=1.0)
KI = 0.004


class Nano3a(NanoRunSeries):
    phrase = "Вводим I"
    formula = "control = uP + uD + integral * ki"
    formula_colors = [("uP", "e"), ("uD", "D"), ("integral * ki", "I")]
    signals = [Signal("uP = err * kp", "uP", "e"), Signal("uI = integral * ki", "uI", "I")]
    y = ("u", -50, 150)
    runs = [
        (dict(PD, ki=0.0), "ki = 0: err ≈ 210"),
        (dict(PD, ki=KI), "ki = 0,004: err → 0"),
    ]

    def make_track(self):
        return ShortTrack()

    def start(self, i):
        return self.track.pose_at(START)


if __name__ == "__main__":
    render(Nano3a, "480p")
