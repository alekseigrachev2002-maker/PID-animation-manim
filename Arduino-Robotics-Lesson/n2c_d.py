"""Урок 2, сцена 2c. Вводим D по алгоритму урока: kp спускаем до «почти» (робот вписывается с 0,07 —
берем 0,06, вылет), затем поднимаем kd, пока снова не впишется. То же место, что в 2b.

По модели с kp = 0,06 робот вписывается уже с kd = 0,5; на экране kd = 1. На длинной дуге D малый kp
не спасает (там err почти не меняется) — поэтому место — вход в поворот с прямой.
"""
from nano_base import NanoRunSeries, render
from nano_track import ShortTrack
from n2b_p import START
from visuals.panel import Signal

KP_ALMOST = 0.06


class Nano2c(NanoRunSeries):
    phrase = "Вводим D"
    formula = "control = err * kp + (err - errOld) * kd"
    formula_colors = [("err * kp", "e"), ("(err - errOld) * kd", "D")]
    signals = [Signal("uP", "uP", "e"), Signal("uD", "uD", "D")]
    y = ("u", -100, 100)
    runs = [
        (dict(kp=KP_ALMOST, kd=0.0), "kp = 0,06: съезд"),
        (dict(kp=KP_ALMOST, kd=1.0), "kd = 1, вписался"),
    ]

    def make_track(self):
        return ShortTrack()

    def start(self, i):
        return self.track.pose_at(START)


if __name__ == "__main__":
    render(Nano2c, "480p")
