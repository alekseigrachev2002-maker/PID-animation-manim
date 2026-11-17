"""Сцена 2b. Спускаем P на повороте «Г» направо; каждый прогон — на одном и том же повороте.

Алгоритм настройки: P, которого хватает на поворот (0.3), — робот поворачивает рывками; спускаемся до P,
при котором робот почти вписывается в поворот (0.1). Дальше — 2c: добавляем D.
Числа — из tune.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenes.base import RunSeries, render  # noqa: E402
from track import Route, Track  # noqa: E402


class Scene2b(RunSeries):
    runs = [
        (0.3, "kp = 0.3, поворачивает рывками"),   # P хватает на поворот, но рулит импульсами
        (0.1, "kp = 0.1, почти"),               # спустились: почти вписывается, вылетает в конце дуги
    ]
    radius = 100.0        # радиус поворота, мм: подобран, чтобы «почти» было kp = 0.1 (на поле 2 углы R = 150)
    t_run = 1.8            # дуга и выход на прямую
    t_ticks = [0, 0.5, 1, 1.5]
    t_slide = 0.7          # возврат одним движением за t_slide + t_turn — влезает в промежуток
    t_turn = 0.5
    return_together = True     # возврат по диагонали: сдвиг и поворот одновременно

    def make_track(self):
        return Track(Route(start=(0, -1500), heading="вверх").forward(1500).mark("поворот")
                     .right(R=self.radius).forward(1500))

    def start(self, i):
        return self.track.pose("поворот", -150)       # на прямой, за 150 мм до дуги


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: видео в videos/Scene2b_<качество>.mp4. Качество: "240p", "480p", "720p", "1080p", "4k".
    render(Scene2b, "480p")
    # render(Scene2b, "1080p")                  # для презентации; все шесть сразу — scenes/lessons.py
