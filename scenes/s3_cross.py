"""Сцена 3. Перекресток: L и R оба ниже порога → стоп. Показываем, как видят перекресток датчики.

Поле 1 без желтых квадратов, то же место, что в сцене 0: вертикаль x = 601.7 под центральным перекрестком
(горизонталь y = 750). Робот стоит у линии, повернутый налево, датчики на белом; его переставляют «рукой» на линию
(линия подъезжает, поворот на месте — правый датчик проходит над линией: один темный — еще не перекресток);
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

X_LINE = 601.7                              # вертикаль поля 1 из сцены 0; центральный перекресток — y = 750


class Scene3(PIDScene):
    # стоит левее линии, повернут налево: датчики на белом; «рука» сдвигает его вправо — датчики линию не пересекают
    stand = (560.0, 470.0, np.radians(90))
    place = (X_LINE, 500.0, 0.0)            # куда ставят: на линию, вверх; до перекрестка датчикам ~140 мм
    ctrl = dict(kp=0.09, kd=1.0)            # регулятор из сцены 2
    t_stand = 0.6                           # стоит на белом, с
    t_slide = 1.0                           # «рука»: линия подъезжает, с
    t_turn = 1.0                            # «рука»: поворот на месте, с

    def panel(self, caption, t_end, t_ticks):
        """Панель сцены: L и R (0…1023), формула-условие, пунктир порога (подпись слева — справа провалы кривых).
        Ее же берет картинка урока (lessons.Scene3Key) — чтобы кадр не расходился с видео."""
        a0, a1 = model.channel_sides(self.sensor)
        L, R = Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")
        panel = GraphPanel(self.fr, self.st, "Перекресток", caption, y=("АЦП", 0, 1023), t_max=t_end * 1.08,
                           t_ticks=t_ticks, signals=[L, R],
                           formula="L < threshold && R < threshold", formula_colors=[("L", "L"), ("R", "R")])
        trs = Controller().threshold
        return panel, panel.plot.hline(trs, f"threshold = {trs:g}", ValueTracker(1.0), side="left")

    def segments(self, track):
        """Отрезки сцены: стоит, линия подъезжает, поворот, езда до перекрестка, торможение."""
        g, s, m = self.geom, self.sensor, Motor()
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
        track = Track(*field1().lines)                     # без желтых квадратов
        segs = self.segments(track)
        ends = np.cumsum([sg.t[-1] for sg in segs])        # конец каждого отрезка на общей шкале
        traj = model.chain(*segs)
        self.show(track, traj)

        panel, limit = self.panel("датчики на белом", traj.t[-1], range(int(traj.t[-1]) + 1))
        self.add(panel, limit, *panel.curves(traj, self.t))
        self.play(panel.intro(), run_time=0.8)          # заголовок выезжает слева

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
    # Кнопка ▶ в PyCharm: видео в videos/Scene3_<качество>.mp4. Качество: "240p", "480p", "720p", "1080p", "4k".
    render(Scene3, "480p")
    # render(Scene3, "1080p")                  # для презентации; все шесть сразу — scenes/lessons.py
