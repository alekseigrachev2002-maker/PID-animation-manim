"""Урок 2, сцена 2c. Вводим D по алгоритму урока: kp спускаем до «почти» (робот вписывается с 0,07 —
берем 0,06, вылет), затем поднимаем kd, пока снова не впишется. То же место, что в 2b (S_TURN_START): за 8 с
робот проходит весь угол R 240.

По модели с kp = 0,06 робот вписывается с kd от 0,3 (0,2 — теряет линию на выходе из угла); на экране kd = 1.
На длинной дуге D малый kp не спасает (там err почти не меняется) — поэтому место — вход в поворот с прямой.
Результат — регулятор уроков 3 и 4 (kp = 0,06, kd = 1). С ним на дуге R 260 после угла err стоит у полки
≈ +700: поворот взят, но со смещением — его убирает И (сцена 3a).
Пояснения короткие, через двоеточие («kd = 0: съезд», «kd = 1: повернул»): формула D длинная, с тире «— повернул»
и «kp = 0,06» в строку не помещаются.
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
        (dict(kp=KP_ALMOST, kd=0.0), "kd = 0: съезд"),
        (dict(kp=KP_ALMOST, kd=1.0), "kd = 1: повернул"),
    ]

    def make_track(self):
        return ShortTrack()

    def start(self, i):
        return self.track.pose_at(START)


if __name__ == "__main__":
    render(Nano2c, "480p")
