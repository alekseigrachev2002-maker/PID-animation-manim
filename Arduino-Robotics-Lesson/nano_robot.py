"""Робот урока 4: робот олимпиадных сцен + ультразвуковой дальномер HC-SR04 на креплении, вид сверху (схема).

Крепление (STL «Body1»): вертикальная пластина 52 × 22 мм с двумя окнами Ø16 под излучатели и лапка снизу —
прикручивается к шасси спереди. Сверху видны: кромка пластины, плата HC-SR04 за ней и два цилиндра излучателей
Ø16, которые выходят вперед через окна. Размеры HC-SR04: плата 45 × 20 мм, излучатели Ø16 через 26 мм, высота 12 мм.

Кубик (сторона 50 мм) и луч дальномера — элементы поля: двигаются вслед за камерой.
"""
from dataclasses import dataclass

import numpy as np
from manim import Polygon, RoundedRectangle, VGroup

from params import Geometry, Sensor, Style
from visuals.robot import Robot

Z_US = 6            # над шасси (4) и метками датчиков (5)
Z_CUBE = 0.4        # на поле, под роботом
Z_BEAM = 0.3


@dataclass(frozen=True)
class Sonar:
    plate_w: float = 52.0       # пластина крепления
    plate_t: float = 3.0        # толщина пластины (вид сверху)
    plate_y: float = 60.0       # задняя грань пластины от оси колес, мм (на шасси спереди)
    pcb_w: float = 45.0         # плата HC-SR04
    pcb_t: float = 1.6
    can_d: float = 16.0         # излучатель
    can_h: float = 12.0         # высота излучателя от платы
    can_x: float = 13.0         # центр излучателя по |x| (26 мм между центрами)
    beam_deg: float = 15.0      # половина угла луча
    plate: str = "#8A94A6"      # пластик крепления
    pcb: str = "#1F4FA8"        # плата (темнее синего L, чтобы не путать с сигналом)
    can: str = "#C9CED6"        # металл излучателя
    mesh: str = "#3A3F48"       # сетка излучателя

    @property
    def front(self):
        """Передний край излучателей (откуда меряется расстояние), мм от оси колес."""
        return self.plate_y - self.pcb_t + self.can_h


def _rect(w, h, cx, cy, r=0.0, **kw):
    return RoundedRectangle(corner_radius=r or 1e-3, width=w, height=h, **kw).move_to([cx, cy, 0])


def sonar_parts(s: Sonar):
    """Дальномер на креплении, мм в системе робота (+y вперед)."""
    pcb = _rect(s.pcb_w, s.pcb_t, 0, s.plate_y - s.pcb_t / 2, fill_color=s.pcb, fill_opacity=1, stroke_width=0)
    plate = _rect(s.plate_w, s.plate_t, 0, s.plate_y + s.plate_t / 2, 0.8, fill_color=s.plate, fill_opacity=1,
                  stroke_width=0)
    cans = VGroup()
    y0 = s.plate_y
    for side in (-1, 1):
        body = _rect(s.can_d, s.can_h, side * s.can_x, y0 + s.can_h / 2 - s.pcb_t, 1.5, fill_color=s.can,
                     fill_opacity=1, stroke_width=0)
        face = _rect(s.can_d - 2.5, 1.6, side * s.can_x, y0 + s.can_h - s.pcb_t - 0.8, 0.6, fill_color=s.mesh,
                     fill_opacity=1, stroke_width=0)
        cans.add(VGroup(body, face))
    return VGroup(pcb, cans, plate)          # пластина поверх цилиндров: они выходят через окна


class SonarRobot(Robot):
    """Робот с дальномером спереди."""

    def __init__(self, fr, geom=Geometry(), st=Style(), sensor=Sensor(), sensors=True, sonar=Sonar()):
        super().__init__(fr, geom, st, sensor, sensors)
        self.sonar = sonar
        part = sonar_parts(sonar).scale(fr.mm, about_point=np.zeros(3)).shift(self.pos)
        part.set_z_index(Z_US)
        self.add(part)


class Cube(VGroup):
    """Кубик на поле (вид сверху), сторона side мм, центр center, поворот angle (рад)."""

    def __init__(self, view, center, side=50.0, angle=0.0, color="#C9A27E", edge="#5A4632"):   # светлое дерево
        super().__init__()
        self.view, self.center, self.side, self.angle = view, np.asarray(center, float), side, angle
        self.top = Polygon(*[[0, 0, 0]] * 4, fill_color=color, fill_opacity=1, stroke_color=edge,
                           stroke_width=view.fr.stroke(2.0))
        self.add(self.top)
        self.set_z_index(Z_CUBE)

    def corners(self):
        h = self.side / 2
        c, s = np.cos(self.angle), np.sin(self.angle)
        loc = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ np.array([[c, s], [-s, c]])
        return self.view.to_screen(loc + self.center)

    def update_state(self, state):
        self.top.set_points_as_corners([*self.corners(), self.corners()[0]])
        return self

    def hit(self, origin, heading, half_deg):
        """Расстояние от origin по курсу heading (рад, как ψ) до кубика, если он в луче ±half_deg; иначе None."""
        d = self.center - np.asarray(origin, float)
        fwd = np.array([-np.sin(heading), np.cos(heading)])
        along = d @ fwd
        if along <= 0:
            return None
        off = abs(d @ np.array([np.cos(heading), np.sin(heading)]))
        # кубик попадает в луч, если его ближний край внутри конуса
        if off - self.side / 2 > along * np.tan(np.radians(half_deg)):
            return None
        return float(along - self.side / 2)


class Beam(VGroup):
    """Луч дальномера: бледный конус ±beam_deg от излучателей до препятствия (кубик) или до max_mm."""

    def __init__(self, view, geom, sonar=Sonar(), cubes=(), max_mm=400.0, color="#D7DCE5"):
        super().__init__()
        self.view, self.geom, self.sonar, self.cubes, self.max_mm = view, geom, sonar, list(cubes), max_mm
        self.cone = Polygon(*[[0, 0, 0]] * 3, fill_color=color, fill_opacity=0.22, stroke_width=0)
        self.add(self.cone)
        self.dist = None
        self.set_z_index(Z_BEAM)

    def update_state(self, state):
        x, y, psi = state["x"], state["y"], state["psi"]
        fwd = np.array([-np.sin(psi), np.cos(psi)])
        o = np.array([x, y]) + fwd * self.sonar.front
        hits = [d for c in self.cubes if (d := c.hit(o, psi, self.sonar.beam_deg)) is not None]
        self.dist = min(hits) if hits else None
        reach = self.dist if self.dist is not None else self.max_mm
        a = np.radians(self.sonar.beam_deg)
        pts = [o]
        for t in np.linspace(-a, a, 9):
            d = np.array([-np.sin(psi + t), np.cos(psi + t)])
            pts.append(o + d * reach / np.cos(t))
        scr = self.view.to_screen(np.array(pts))
        self.cone.set_points_as_corners([*scr, scr[0]])
        return self
