"""Сцена 2c. Вводим D: тот же поворот «Г» направо, тот же kp = 0.12, что почти вписывался в 2b.

kd = 0 — вылет в конце дуги (повтор из 2b); kd = 1 — проходит. На графике составляющие управления:
uP = err * kp и uD = (err - errOld) * kd — на входе в поворот uD срабатывает первым.
Период регулятора 10 мс (delay(10) в скетче): kd здесь — то же число, что в коде.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenes.base import RunSeries, render  # noqa: E402
from track import Route, Track  # noqa: E402
from visuals.panel import Signal  # noqa: E402


class Scene2c(RunSeries):
    phrase = "Вводим D"
    formula = "control = err * kp + (err - errOld) * kd"          # строка из скетча
    formula_colors = [("err * kp", "e"), ("(err - errOld) * kd", "D")]
    signals = [Signal("uP", "uP", "e"), Signal("uD", "uD", "D")]   # что это — показывает формула в шапке теми же цветами
    y = ("u", -150, 150)            # uD при kd = 1 — до ±120
    runs = [
        (dict(kp=0.12, kd=0.0), "kd = 0, вылет"),
        (dict(kp=0.12, kd=1.0), "kd = 1, прошел"),
    ]
    radius = 150.0        # тот же поворот, что в 2b
    t_run = 3.4            # дуга и выход на прямую
    t_ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    t_slide = 1.2

    def make_track(self):
        return Track(Route(start=(0, -1500), heading="вверх").forward(1500).mark("поворот")
                     .right(R=self.radius).forward(1500))

    def start(self, i):
        return self.track.pose("поворот", -350)       # на прямой, за 350 мм до дуги


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm. Качество: "240p", "480p", "720p", "1080p", "4k"; still=True — только последний кадр.
    # render(Scene2c, "1080p", still=True)
    render(Scene2c)
