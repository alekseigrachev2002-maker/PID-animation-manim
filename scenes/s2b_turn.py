"""Сцена 2b. Подбор P на повороте «Г» направо; каждый прогон — на одном и том же повороте.

Малый P держит линию на прямой, но на повороте не успевает развернуть робота: датчик проскакивает линию
за ~0.1 с, err даёт один короткий импульс, робот уезжает прямо. Достаточный P поворот проходит.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenes.base import RunSeries, render  # noqa: E402
from track import Route, Track  # noqa: E402


class Scene2b(RunSeries):
    runs = [
        (0.1, "kp = 0.1, поворота не хватает"),
        (0.3, "kp = 0.3, поворот пройден"),
    ]
    radius = 100.0        # радиус поворота, мм — заглушка до замера трассы
    t_run = 2.6
    t_ticks = [0, 0.5, 1, 1.5, 2, 2.5]
    t_slide = 1.2

    def make_track(self):
        return Track(Route(start=(0, -1500), heading="вверх").forward(1500).mark("поворот")
                     .right(R=self.radius).forward(1500))

    def start(self, i):
        return self.track.pose("поворот", -350)       # на прямой, за 350 мм до дуги


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p:
    render(Scene2b, "high_quality", still=True)
    # render(Scene2b, "high_quality")
