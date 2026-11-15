"""Сцена 2b. Спускаем P на повороте «Г» направо; каждый прогон — на одном и том же повороте.

Алгоритм настройки: в 2a подняли P до перерегулирования (1); здесь на повороте с ним робот вписывается, но дергается,
и спускаемся до P, при котором робот почти вписывается в поворот (0.09). Дальше — 2c: добавляем D.
Числа — из tune.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenes.base import RunSeries, render  # noqa: E402
from track import Route, Track  # noqa: E402


class Scene2b(RunSeries):
    runs = [
        (1.0, "kp = 1, вписался, но дергается"),  # P из конца 2a: поворот проходит, но рывками
        (0.09, "kp = 0.09, почти"),               # спустились: почти вписывается, вылетает в конце дуги
    ]
    radius = 150.0        # радиус поворота, мм — углы поля 2400×1500 с плавной линией
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
    render(Scene2b, "1080p", still=True)
    # render(Scene2b, "1080p")
