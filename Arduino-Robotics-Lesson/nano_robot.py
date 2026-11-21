"""Робот урока 4: робот олимпиадных сцен + ультразвуковой дальномер HC-SR04 на креплении, вид сверху (схема).

Крепление (STL «Body1», белый пластик): лапка с двумя отверстиями прикручена к шасси в отверстия (±8; 70) —
Chassis.dxf (винты белые — не рисуются), вертикальная пластина 52 × 22 мм с окнами Ø16 стоит на переднем крае шасси. Плата HC-SR04 вставлена
в паз за пластиной, излучатели выходят вперед через окна. Сверху видны: лапка, «уши» паза, кромка платы,
пластина и излучатели. HC-SR04: плата 45 × 20 мм, излучатели Ø16 через 26 мм, высота 12 мм.

Кубик (сторона 50 мм) — элемент поля: двигается вслед за камерой. Луч дальномера не рисуется: sonar_distance
считает, что покажет датчик.
"""
from dataclasses import dataclass

import numpy as np
from manim import Polygon, RoundedRectangle, VGroup

from params import Geometry, Sensor, Style
from visuals.robot import Robot

Z_US = 6            # над шасси (4) и метками датчиков (5)
Z_CUBE = 0.4        # на поле, под роботом


@dataclass(frozen=True)
class Sonar:
    """Дальномер на креплении Body1 (белый пластик), мм в системе робота: +y вперед, 0 — середина оси колес.

    Крепление (по STL): лапка 22 × 11 мм с двумя отверстиями (±8,7; −8,7 от передней грани пластины) лежит на шасси
    и прикручена в отверстия шасси (±8; 70) (Chassis.dxf); вертикальная пластина 52 мм с окнами Ø16 стоит на переднем
    крае шасси, по бокам — «уши» паза, в паз за пластиной вставлена плата HC-SR04, излучатели выходят в окна.
    """
    hole_y: float = 70.0        # отверстия шасси под лапку (Chassis.dxf: окружности Ø2 в (±8; 70))
    hole_x: float = 8.7         # отверстия лапки по |x| (STL)
    hole_back: float = 8.7      # от отверстий лапки до задней грани пластины крепления, мм (STL: −8,7)
    plate_w: float = 52.0       # пластина крепления
    plate_t: float = 2.0        # толщина пластины (вид сверху)
    ear_x: tuple = (17.0, 26.0) # «уши» паза под плату по |x|
    ear_d: float = 4.0          # глубина паза за пластиной
    tab_w: float = 22.0         # лапка
    tab_d: float = 11.0
    pcb_w: float = 45.0         # плата HC-SR04
    pcb_t: float = 1.6
    can_d: float = 16.0         # излучатель
    can_h: float = 12.0         # высота излучателя от платы
    can_x: float = 13.0         # центр излучателя по |x| (26 мм между центрами)
    beam_deg: float = 15.0      # половина угла луча
    mount: str = "#F4F4F2"      # белый пластик крепления
    mount_edge: str = "#9AA3AF"
    pcb: str = "#1F4FA8"        # плата (темнее синего L, чтобы не путать с сигналом)
    can: str = "#C9CED6"        # металл излучателя
    mesh: str = "#3A3F48"       # сетка излучателя

    @property
    def plate_y(self):
        """Задняя грань пластины крепления (она же — передняя грань платы), мм от оси колес."""
        return self.hole_y + self.hole_back

    @property
    def front(self):
        """Передний край излучателей (откуда меряется расстояние), мм от оси колес."""
        return self.plate_y + self.can_h


def _rect(w, h, cx, cy, r=0.0, **kw):
    return RoundedRectangle(corner_radius=r or 1e-3, width=w, height=h, **kw).move_to([cx, cy, 0])


def sonar_parts(s: Sonar, edge_w):
    """Крепление с дальномером, мм в системе робота (+y вперед). edge_w — толщина контура белого пластика."""
    white = dict(fill_color=s.mount, fill_opacity=1, stroke_color=s.mount_edge, stroke_width=edge_w)
    y0 = s.plate_y
    tab = _rect(s.tab_w, s.tab_d, 0, y0 - s.tab_d / 2, 1.5, **white)
    pcb = _rect(s.pcb_w, s.pcb_t, 0, y0 - s.pcb_t / 2, fill_color=s.pcb, fill_opacity=1, stroke_width=0)
    ears = VGroup(*[_rect(s.ear_x[1] - s.ear_x[0], s.ear_d, sx * sum(s.ear_x) / 2, y0 - s.ear_d / 2, 0.8, **white)
                    for sx in (-1, 1)])
    cans = VGroup()
    for side in (-1, 1):
        body = _rect(s.can_d, s.can_h, side * s.can_x, y0 + s.can_h / 2, 1.5, fill_color=s.can, fill_opacity=1,
                     stroke_width=0)
        face = _rect(s.can_d - 2.5, 1.6, side * s.can_x, y0 + s.can_h - 0.8, 0.6, fill_color=s.mesh,
                     fill_opacity=1, stroke_width=0)
        cans.add(VGroup(body, face))
    plate = _rect(s.plate_w, s.plate_t, 0, y0 + s.plate_t / 2, 0.8, **white)
    return VGroup(tab, pcb, ears, cans, plate)    # пластина поверх излучателей (они выходят через окна); винты не рисуем


class SonarRobot(Robot):
    """Робот с дальномером спереди."""

    def __init__(self, fr, geom=Geometry(), st=Style(), sensor=Sensor(), sensors=True, sonar=Sonar()):
        super().__init__(fr, geom, st, sensor, sensors)
        self.sonar = sonar
        part = sonar_parts(sonar, fr.stroke(0.8)).scale(fr.mm, about_point=np.zeros(3)).shift(self.pos)
        part.set_z_index(Z_US)
        self.add(part)


class Cube(VGroup):
    """Кубик на поле (вид сверху), сторона side мм, центр center, поворот angle (рад)."""

    def __init__(self, view, center, side=50.0, angle=0.0, color="#22C55E", edge="#166534"):   # зеленый кубик
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
        return cube_hit(self.center, self.side, origin, heading, half_deg)


def cube_hit(center, side, origin, heading, half_deg):
    """Расстояние от origin по курсу heading (рад, как ψ) до кубика (центр center, сторона side), если его ближний
    край внутри луча ±half_deg; иначе None."""
    d = np.asarray(center, float) - np.asarray(origin, float)
    fwd = np.array([-np.sin(heading), np.cos(heading)])
    along = d @ fwd
    if along <= 0:
        return None
    off = abs(d @ np.array([np.cos(heading), np.sin(heading)]))
    if off - side / 2 > along * np.tan(np.radians(half_deg)):
        return None
    return float(along - side / 2)


def sonar_distance(pose, cubes, sonar=Sonar(), max_mm=600.0):
    """Что покажет дальномер в позе pose = (x, y, ψ): расстояние от излучателей до ближнего кубика в луче
    ±beam_deg, мм; нет кубика в луче — max_mm (на графике — верх шкалы). cubes — [(центр, сторона)].
    Луч на экране не рисуется."""
    x, y, psi = pose
    o = np.array([x, y]) + np.array([-np.sin(psi), np.cos(psi)]) * sonar.front
    hits = [d for c, side in cubes if (d := cube_hit(c, side, o, psi, sonar.beam_deg)) is not None]
    return min(hits + [max_mm])
