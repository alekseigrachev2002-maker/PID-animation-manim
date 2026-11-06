"""Робот сверху. Строится в мм (система робота: +y вперёд), на экран — через Frame (масштаб и точка оси)."""
import numpy as np
import math

from manim import UL, Arc, Circle, Difference, Rectangle, RoundedRectangle, Union, VGroup, VMobject

from params import Geometry, Style

N_TREAD = 18          # шевронов ёлочки по окружности колеса
Z = dict(wheel=1, tread=2, board=3, chassis=4)


def _rect(w, h, cx, cy, r=0.0, **kw):
    m = RoundedRectangle(corner_radius=r, width=w, height=h, **kw) if r else Rectangle(width=w, height=h, **kw)
    return m.move_to([cx, cy, 0])


def chassis(g: Geometry, st: Style, edge_w):
    x0, x1 = g.cutout_x
    cuts = [_rect(x1 - x0, g.cutout_h, s * (x0 + x1) / 2, 0, g.cutout_r) for s in (-1, 1)]
    body = Difference(Circle(radius=g.chassis_r), Union(*cuts))
    body.set_fill(color=[st.chassis, st.chassis_hi], opacity=st.chassis_opacity).set_sheen_direction(UL)
    body.set_stroke(color=st.chassis_edge, width=edge_w)
    # блик глянца: дуга у края с верхней левой стороны
    glare = Arc(radius=g.chassis_r - 7, start_angle=math.radians(105), angle=math.radians(50))
    glare.set_stroke(color=st.glare, width=4 * edge_w, opacity=0.18)
    return VGroup(body, glare)


def wheels(g: Geometry, st: Style):
    return VGroup(*[_rect(g.wheel_w, g.wheel_d, s * g.wheel_x, 0, 2.5, fill_color=st.wheel, fill_opacity=1,
                          stroke_width=0) for s in (-1, 1)])


def boards(g: Geometry, st: Style):
    front = g.mount_y + g.board_front_to_mount
    out = VGroup()
    for s, color in ((-1, st.L), (1, st.R)):
        board = _rect(g.board_w, g.board_len, s * g.sensor_x, front - g.board_len / 2, 0.8,
                      fill_color=color, fill_opacity=1, stroke_width=0)
        hole = Circle(radius=1.5, fill_color=st.field, fill_opacity=1, stroke_width=0).move_to(
            [s * g.sensor_x, front - 2.5, 0])
        out.add(VGroup(board, hole))
    return out


def tread_points(g: Geometry, side, phase):
    """Шевроны видимой половины колеса: [(точки 3×2 мм, видимость 0…1)].

    Шеврон на угле φ от верхней точки колеса виден сверху при cos φ > 0 и стоит на y = R·sin φ.
    Ёлочка смотрит вершиной вперёд; при езде вперёд верх колеса движется вперёд относительно корпуса.
    """
    R, hw = g.wheel_d / 2, g.wheel_w / 2 - 1.2
    xc = side * g.wheel_x
    out = []
    for k in range(N_TREAD):
        phi = phase + 2 * np.pi * k / N_TREAD
        c, y = np.cos(phi), R * np.sin(phi)
        vis = max(c, 0.0) if abs(y) < R - 2.5 else 0.0
        pts = np.array([[xc - hw, y - 2.2], [xc, y + 1.0], [xc + hw, y - 2.2]])
        out.append((pts, vis))
    return out


class Robot(VGroup):
    """Робот на экране: середина оси колёс в fr.anchor. set_pose(psi), set_wheel_phase(sL, sR)."""

    def __init__(self, fr, geom=Geometry(), st=Style()):
        super().__init__()
        self.fr, self.geom, self.st = fr, geom, st
        self.psi = 0.0
        self.treads = VGroup(*[VMobject(stroke_color=st.tread) for _ in range(2 * N_TREAD)])
        # оптопара TCRT стоит на нижней стороне платы и сверху не видна — не рисуем
        parts = dict(wheel=wheels(geom, st), tread=self.treads, board=boards(geom, st),
                     chassis=chassis(geom, st, fr.stroke(1.2)))
        for name, m in parts.items():
            if name != "tread":
                m.scale(fr.mm, about_point=np.zeros(3)).shift(fr.anchor)
            m.set_z_index(Z[name])
            self.add(m)
        self.set_wheel_phase(0.0, 0.0)

    def _to_screen(self, pts_mm):
        c, s = np.cos(self.psi), np.sin(self.psi)
        p = np.asarray(pts_mm) @ np.array([[c, s], [-s, c]]) * self.fr.mm
        return np.column_stack([p, np.zeros(len(p))]) + self.fr.anchor

    def set_pose(self, psi):
        self.rotate(psi - self.psi, about_point=self.fr.anchor)
        self.psi = psi
        return self

    def set_wheel_phase(self, sL, sR):
        """sL, sR — путь колёс в мм; фаза протектора = s / R."""
        R = self.geom.wheel_d / 2
        i = 0
        for side, s in ((-1, sL), (1, sR)):
            for pts, vis in tread_points(self.geom, side, s / R):
                m = self.treads[i]
                m.set_points_as_corners(self._to_screen(pts))
                m.set_stroke(width=self.fr.stroke_mm(1.6) * vis, opacity=0.9 if vis > 0 else 0)
                i += 1
        return self
