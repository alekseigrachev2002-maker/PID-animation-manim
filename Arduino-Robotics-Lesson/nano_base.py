"""Основа сцен нового курса — тонкий слой над движком (scenes/base.py): датчики Short Track и камера,
повернутая так, что на старте сцены робот едет снизу вверх. Все остальное — в движке: подсцены MOTION,
ускорение (fast, advance_fast), «рука» (hand_to), серия прогонов по 8 с (MotionRunSeries), робот с дальномером,
кубики на поле. Импорт этого модуля добавляет в путь корень репозитория — в сценах он первый.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from nano_track import SENSOR  # noqa: E402
from scenes.base import MotionRunSeries, PIDScene, render  # noqa: E402

__all__ = ["ROOT", "NanoScene", "NanoRunSeries", "render"]


class NanoScene(PIDScene):
    """Сцена нового курса: датчики (белое 900, черное 200, линия 20 мм), на старте робот едет вверх."""
    sensor = SENSOR
    rot = "start"


class NanoRunSeries(NanoScene, MotionRunSeries):
    """Серия прогонов по 8 с (MotionRunSeries) с датчиками и камерой нового курса."""
