"""Сцена 2a. Подбор P на прямой: нет регуляции → едет по линии → перерегулирование.

Прогоны идут подряд по одной линии. Между ними — возврат: робот всегда встаёт точно на центр линии
(если съехал — линия подъезжает), затем поворачивается на стартовый угол. На центр — всегда: иначе поворот
на 15° при смещении в мёртвой зоне может увести оба датчика по одну сторону, и прогон начнётся без линии.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scenes.base import RunSeries, render  # noqa: E402
from track import Route, Track  # noqa: E402


class Scene2a(RunSeries):
    runs = [
        (0.0, "kp = 0, нет регуляции"),
        (0.3, "kp = 0.3, едет по линии"),
        (0.6, "kp = 0.6, перерегулирование"),
    ]
    psi0_deg = 15.0       # стартовый поворот относительно линии

    def make_track(self):
        return Track(Route(start=(0, -300), heading="вверх").forward(20000))

    def start(self, i):
        y = 0.0 if i == 0 else self.end_pose()[1]
        return 0.0, y, np.radians(self.psi0_deg)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p:
    render(Scene2a, "high_quality", still=True)
    # render(Scene2a, "high_quality")
