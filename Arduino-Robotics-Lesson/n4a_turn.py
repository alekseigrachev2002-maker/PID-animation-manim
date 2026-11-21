"""Урок 4, сцена 4a — МАКЕТ (кадры, без движения). Поворот на перекрестке к кубику.

Задача: робот едет по линии от старта (линия старт/финиш на верхней внутренней прямой) по часовой стрелке,
доезжает до главного перекрестка X, тормозит, поворачивает налево или направо — туда, где кубик 5 см, — и едет
до финиша. Направо — через S-изгиб и левую петлю к финишу на внутренней прямой (2,1 м), налево — по дуге R 700
и внешней петле к финишу на внешней прямой (2,7 м).

Макет: три кадра — подъезд к X, робот на X повернут направо (дальномер видит кубик), робот уехал налево —
кубик стоит на линии правой ветки и загораживает ее.
Правая панель узкая (267 мм поля при обычном масштабе), а ветка после поворота уходит вбок — поэтому здесь
камера неподвижна и масштаб мельче (SCALE px/мм вместо 1,6): в кадре весь перекресток, подъезд и кубик.
Кубик стоит на линии правой ветки, CUBE_ALONG мм от X (центр кубика).
Кадры — python Arduino-Robotics-Lesson/n4a_turn.py (720p, renders/Nano4a*_720p.png).
"""
from nano_base import NanoScene, render  # первым: добавляет в путь корень репозитория

from params import Layout
from visuals.layout import Frame

import numpy as np
from manim import ValueTracker

import model
from nano_robot import Beam, Cube, Sonar, SonarRobot
from nano_track import ShortTrack
from visuals.panel import GraphPanel, Signal

S_X = 4847.0          # перекресток X по пути от старта (по часовой)
S_OTHER = 617.0       # та же точка X на другой линии (по ней — ветки направо и налево)
CUBE_ALONG = 150.0    # кубик: на линии правой ветки, от X до центра кубика, мм
SCALE = 1.3           # px/мм эталонного кадра 1280×720 (в остальных сценах 1,6)
SHIFT = 60.0          # центр кадра: от X в сторону правой ветки, мм
APPROACH = 180.0      # кадр «подъезд»: столько мм до X


def cube_pose(track):
    """Центр и поворот кубика: на линии правой ветки, грань поперек линии."""
    x, y, psi = track.route.pose_at(S_OTHER + CUBE_ALONG)
    return np.array([x, y]), psi


class Nano4aMock(NanoScene):
    robot_cls = SonarRobot
    caption = ""
    stage = "approach"

    def setup(self):
        super().setup()
        self.fr = Frame(Layout(px_per_mm=SCALE))

    def pose(self, track):
        if self.stage == "approach":
            return track.route.pose_at(S_X - APPROACH)
        x, y, _ = track.route.pose_at(S_X)
        if self.stage == "turned":
            return (x, y, track.route.pose_at(S_OTHER)[2])          # на X, курс — по правой ветке
        x, y, psi = track.route.pose_at(S_OTHER - 110)                # уехал по левой ветке (против хода линии)
        return (x, y, psi + np.pi)

    def field_items(self, view):
        center, angle = cube_pose(self.track_)
        self.cube = Cube(view, center, angle=angle)
        self.beam = Beam(view, self.geom, Sonar(), cubes=[self.cube])
        return [self.beam, self.cube]

    def construct(self):
        track = self.track_ = ShortTrack()
        p = self.pose(track)
        traj = model.prescribed(track, np.array([0.0, 0.01]), *p, sensor=self.sensor)
        x, y, psi = track.route.pose_at(S_OTHER)
        center = np.array([x, y]) + np.array([-np.sin(psi), np.cos(psi)]) * SHIFT
        self.show(track, traj, camera="fixed", center=center, anchor_px=(1066.5, 360.0),
                  rot=-track.route.pose_at(S_X - APPROACH)[2])        # на подъезде робот едет вверх
        dist = self.beam.dist
        caption = self.caption.format(d=(dist or 0) / 10)
        panel = GraphPanel(self.fr, self.st, "Кубик на перекрестке", caption, y=("dist, см", 0, 100), t_max=3.2,
                           t_ticks=[0, 1, 2, 3], signals=[Signal("dist — дальномер", "L", "#D7DCE5")],
                           formula="if (dist < 30) направо; else налево;",
                           formula_colors=[("dist", "#D7DCE5")])
        self.add(panel, panel.plot.hline(30, "30 см", ValueTracker(1.0), side="left"))
        self.wait(0.1, frozen_frame=False)


class Nano4aApproach(Nano4aMock):
    stage = "approach"
    caption = "едем к перекрестку"


class Nano4aTurned(Nano4aMock):
    stage = "turned"
    caption = "на X: dist = {d:.0f} см"


class Nano4aAway(Nano4aMock):
    stage = "away"
    caption = "едем налево"


if __name__ == "__main__":
    for cls in (Nano4aApproach, Nano4aTurned, Nano4aAway):
        render(cls, "720p", still=True, preview=False)
