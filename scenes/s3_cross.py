"""Сцена 3. Перекрёсток: L и R оба ниже порога → стоп. Показываем, как видят перекрёсток датчики.

Поле 1 без жёлтых квадратов, то же место, что в сцене 0: вертикаль x = 601.7 под центральным перекрёстком
(горизонталь y = 750). Робот стоит у линии, повёрнутый налево, датчики на белом; его переставляют «рукой» на линию
(линия подъезжает, поворот на месте — правый датчик проходит над линией: один тёмный — ещё не перекрёсток);
он едет по линии регулятором из сцены 2 и на тике, где оба датчика ниже порога, — motors(0, 0) и докатывается.
График L и R — через всю сцену, вместе с «рукой». Камера едет за роботом.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import ValueTracker  # noqa: E402

import model  # noqa: E402
from params import Controller, Motor  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from track import Track, field1  # noqa: E402
from visuals.panel import GraphPanel, Signal  # noqa: E402

X_LINE = 601.7                              # вертикаль поля 1 из сцены 0; центральный перекрёсток — y = 750


class Scene3(PIDScene):
    # стоит левее линии, повёрнут налево: датчики на белом; «рука» сдвигает его вправо — датчики линию не пересекают
    stand = (560.0, 470.0, np.radians(90))
    place = (X_LINE, 500.0, 0.0)            # куда ставят: на линию, вверх; до перекрёстка датчикам ~140 мм
    ctrl = dict(kp=0.13, kd=0.5)            # регулятор из сцены 2
    delay = 0.020                           # как в сценах 2
    t_stand = 0.6                           # стоит на белом, с
    t_slide = 1.0                           # «рука»: линия подъезжает, с
    t_turn = 1.0                            # «рука»: поворот на месте, с
    speed = 0.33                            # езда — втрое медленнее, как в сценах 2; «рука» — в реальном времени

    def segments(self, track):
        """Отрезки сцены: стоит, линия подъезжает, поворот, езда до перекрёстка, торможение."""
        g, s, m = self.geom, self.sensor, Motor(delay=self.delay)
        x, y, psi = self.stand
        tx, ty, tpsi = self.place
        stand = model.prescribed(track, np.arange(0, self.t_stand + 1e-9, 0.005), x, y, psi, g, s)
        slide = model.move(track, (x, y, psi), (tx, ty, psi), self.t_slide, roll=False, geom=g, sensor=s)
        turn = model.move(track, (tx, ty, psi), (tx, ty, tpsi), self.t_turn, geom=g, sensor=s)
        run = model.cached_simulate(track, self.place, ctrl=Controller(**self.ctrl), motor=m, t_end=3.0,
                                    stop_below=Controller().threshold)
        stop = model.brake(track, run, geom=g, sensor=s, motor=m)
        return [stand, slide, turn, run, stop]

    def construct(self):
        st, sens = self.st, self.sensor
        track = Track(*field1().lines)                     # без жёлтых квадратов
        segs = self.segments(track)
        ends = np.cumsum([sg.t[-1] for sg in segs])        # конец каждого отрезка на общей шкале
        traj = model.chain(*segs)
        self.show(track, traj)

        a0, a1 = model.channel_sides(sens)
        L, R = Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")
        panel = GraphPanel(self.fr, st, "Перекрёсток", "датчики на белом", y=("АЦП", 0, 1023),
                           t_max=traj.t[-1] * 1.08, t_ticks=range(int(traj.t[-1]) + 1), signals=[L, R])
        trs = Controller().threshold
        limit = panel.plot.hline(trs, f"threshold = {trs:g}", ValueTracker(1.0))
        self.add(panel, limit, *panel.curves(traj, self.t))
        self.play(panel.set_formula("L < threshold && R < threshold", [("L", "L"), ("R", "R")]), run_time=0.8)

        self.advance(ends[0], speed=1.0)
        self.play(panel.set_caption("ставим на линию"), run_time=0.6)
        self.advance(ends[2], speed=1.0)
        self.play(panel.set_caption("едем по линии"), run_time=0.6)
        self.advance(ends[3])
        # стоп: подпись меняется в начале докатывания (0.6 с на экране) — время не замирает на ходу
        self.advance(ends[3] + 0.6 * self.speed, panel.set_caption("оба ниже порога: стоп"))
        self.advance(ends[4])
        self.wait(1.5)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm. Качество: "240p", "480p", "720p", "1080p", "4k"; still=True — только последний кадр.
    render(Scene3, "240p")
    # render(Scene3, "1080p")
