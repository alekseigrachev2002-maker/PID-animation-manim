"""Урок 3, сцена 3a. Смещение на дуге и И.

Робот въезжает с верхней прямой в правую внутреннюю петлю (по часовой — поворот направо, R 240, дальше R 260);
старт — то же место, что в 2b и 2c (S_TURN_START, за 12 см до дуги — без стартового выброса uP). Регулятор
уроков 3 и 4 — kp = 0,1, kd = 1 (kp с запасом над «почти» 0,06 из 2c: с ним И в итоговой сцене не раскачивает
робота на прямых). ПД едет со смещением: чтобы поворачивать, нужен постоянный control, а его дает только постоянная
ошибка — err стоит полкой ≈ +410, uP ≈ +41. И копит ошибку: uI растет и берет поворот на себя, err к концу прогона
≈ 0 (−11).

ki проверен по модели при kp 0,1: 0,003 — err к концу −3, но медленнее; 0,005 — err в середине прогона 131,
к концу −11; 0,008 — к концу снова +25. Взято 0,005.
"""
from nano_base import NanoRunSeries, render
from nano_track import S_TURN_START, ShortTrack
from visuals.panel import Signal

START = S_TURN_START
PD = dict(kp=0.1, kd=1.0)    # регулятор уроков 3 и 4 (kp с запасом над «почти» 0,06 из 2c)
KI = 0.005            # err → 0 за прогон; в 4a с kp 0,1 — 13 переходов через линию на прямых вместо 24 при kp 0,06


class Nano3a(NanoRunSeries):
    phrase = "Вводим I"
    formula = "control = uP + uD + uI"                              # как в сцене 4a
    formula_colors = [("uP", "e"), ("uD", "D"), ("uI", "I")]
    signals = [Signal("uP = err * kp", "uP", "e"), Signal("uI = integral * ki", "uI", "I")]
    y = ("u", -25, 100)             # uP до ~42 (полка err ≈ 700), uI до ~57
    runs = [
        (dict(PD, ki=0.0), "ki = 0"),
        (dict(PD, ki=KI), f"ki = {KI:g}: err → 0".replace(".", ",")),
    ]

    def make_track(self):
        return ShortTrack()

    def start(self, i):
        return self.track.pose_at(START)


if __name__ == "__main__":
    render(Nano3a, "480p")
