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
from scenes.base import HAND_V, HAND_W, MotionRunSeries, PIDScene, render  # noqa: E402

__all__ = ["ROOT", "NanoScene", "NanoRunSeries", "render"]

HAND_SPEED = 2.0   # во сколько раз «рука» быстрее базовой (база: 150 мм/с, 120°/с); 1,0 — как в олимпиадных сценах
T_AXIS = 2.5       # шкала времени графика в сериях прогонов (2b, 2c, 3a), с: одна — графики можно сравнивать


class NanoScene(PIDScene):
    """Сцена нового курса: датчики (белое 900, черное 200, линия 20 мм), на старте робот едет вверх;
    «рука» — в HAND_SPEED раз быстрее олимпиадной."""
    sensor = SENSOR
    rot = "start"
    hand_v = HAND_V * HAND_SPEED
    hand_w = HAND_W * HAND_SPEED


class NanoRunSeries(NanoScene, MotionRunSeries):
    """Серия прогонов по 8 с (MotionRunSeries) с датчиками и камерой нового курса. Ось времени — 0…T_AXIS
    с делениями через 0,5 с во всех сериях (прогон — 8 с видео, в модели ~1,9 с)."""

    def panel(self, t_max):
        return super().panel(T_AXIS * 1.04)          # деления 0…2,5 (np.arange в движке); запас справа — под подпись

    def plan(self):
        runs = super().plan()
        if max(r.t[-1] for r in runs) > T_AXIS:
            raise ValueError(f"{type(self).__name__}: прогон длиннее шкалы T_AXIS = {T_AXIS} с")
        return runs
