"""Робот сверху. Строится в мм (система робота: +y вперёд); на экран ставится методом place().

Части: колёса с протектором, оси (цвет стороны), шасси, платы датчиков с метками (флажок sensors).
Оптопара TCRT стоит на нижней стороне платы и сверху не видна — не рисуем.
"""
import numpy as np
from manim import WHITE, Circle, Difference, ManimColor, Rectangle, RoundedRectangle, Union, VGroup, VMobject, \
    interpolate_color

from params import Geometry, Sensor, Style

HALO_R = (11.0, 8.0, 5.5)   # ореол метки: круги внахлёст, к краю свечение слабее
N_TREAD = 18                # шевронов ёлочки по окружности колеса
Z = dict(wheel=1, tread=2, axle=2.5, board=3, chassis=4, marker=5)


def _rect(w, h, cx, cy, r=0.0, **kw):
    m = RoundedRectangle(corner_radius=r, width=w, height=h, **kw) if r else Rectangle(width=w, height=h, **kw)
    return m.move_to([cx, cy, 0])


def chassis(g: Geometry, st: Style, edge_w):
    """Шасси: круг с вырезами под колёса."""
    x0, x1 = g.cutout_x
    cuts = [_rect(x1 - x0, g.cutout_h, s * (x0 + x1) / 2, 0, g.cutout_r) for s in (-1, 1)]
    body = Difference(Circle(radius=g.chassis_r), Union(*cuts))
    return body.set_fill(st.chassis, 1).set_stroke(st.chassis_edge, edge_w)


def wheels(g: Geometry, st: Style):
    return VGroup(*[_rect(g.wheel_w, g.wheel_d, s * g.wheel_x, 0, 2.5, fill_color=st.wheel, fill_opacity=1,
                          stroke_width=0) for s in (-1, 1)])


def axles(g: Geometry, st: Style):
    """Видимые оси: от края выреза в акриле до внутренней стороны колеса; цвет — сторона робота."""
    x0, x1 = g.cutout_x[0], g.wheel_x - g.wheel_w / 2
    return VGroup(*[_rect(x1 - x0, g.axle_w, s * (x0 + x1) / 2, 0, fill_color=c, fill_opacity=1, stroke_width=0)
                    for s, c in ((-1, st.L), (1, st.R))])


def board_colors(st: Style, sensor: Sensor):
    """Цвета плат [левая, правая] = цвет сигнала, который с неё идёт (L на A0, R на A1)."""
    return (st.R, st.L) if sensor.swapped else (st.L, st.R)


def boards(g: Geometry, st: Style, sensor: Sensor):
    front = g.mount_y + g.board_front_to_mount
    out = VGroup()
    for s, color in zip((-1, 1), board_colors(st, sensor)):
        board = _rect(g.board_w, g.board_len, s * g.sensor_x, front - g.board_len / 2, 0.8,
                      fill_color=color, fill_opacity=1, stroke_width=0)
        hole = Circle(radius=1.5, fill_color=st.field, fill_opacity=1, stroke_width=0).move_to(
            [s * g.sensor_x, front - 2.5, 0])
        out.add(VGroup(board, hole))
    return out


def markers(g: Geometry, st: Style, sensor: Sensor):
    """Метки оптического центра на платах [левая, правая]: ореол цвета сигнала и ядро."""
    out = VGroup()
    for s, color in zip((-1, 1), board_colors(st, sensor)):
        c = [s * g.sensor_x, g.sensor_y, 0]
        halo = VGroup(*[Circle(radius=r, stroke_width=0).set_fill(color, 0).move_to(c) for r in HALO_R])
        core = Circle(radius=3.2, stroke_width=0).set_fill(marker_core(color, 0.0), 1).move_to(c)
        out.add(VGroup(halo, core))
    return out


def marker_core(color, level):
    """Цвет ядра метки: тёмный оттенок платы (белое под датчиком) → почти белый (чёрное)."""
    dim = interpolate_color(ManimColor(color), ManimColor("#000000"), 0.45)
    return interpolate_color(dim, WHITE, 0.85 * level)


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
        out.append((np.array([[xc - hw, y - 2.2], [xc, y + 1.0], [xc + hw, y - 2.2]]), vis))
    return out


class Robot(VGroup):
    """place(точка экрана, ψ) — поставить середину оси в точку и повернуть; set_wheel_phase, set_signals."""

    def __init__(self, fr, geom=Geometry(), st=Style(), sensor=Sensor(), sensors=True):
        super().__init__()
        self.fr, self.geom, self.st, self.sensor = fr, geom, st, sensor
        self.pos, self.psi = fr.anchor.copy(), 0.0
        self.treads = VGroup(*[VMobject(stroke_color=st.tread) for _ in range(2 * N_TREAD)])
        self.markers = markers(geom, st, sensor) if sensors else VGroup()
        parts = dict(wheel=wheels(geom, st), tread=self.treads, axle=axles(geom, st),
                     chassis=chassis(geom, st, fr.stroke(1.2)))
        if sensors:
            parts.update(board=boards(geom, st, sensor), marker=self.markers)
        for name, m in parts.items():
            if name != "tread":
                m.scale(fr.mm, about_point=np.zeros(3)).shift(self.pos)
            m.set_z_index(Z[name])
            self.add(m)
        self.set_wheel_phase(0.0, 0.0)

    def _to_screen(self, pts_mm):
        c, s = np.cos(self.psi), np.sin(self.psi)
        p = np.asarray(pts_mm) @ np.array([[c, s], [-s, c]]) * self.fr.mm
        return np.column_stack([p, np.zeros(len(p))]) + self.pos

    def place(self, point, psi):
        point = np.asarray(point, float)
        self.shift(point - self.pos)
        self.rotate(psi - self.psi, about_point=point)
        self.pos, self.psi = point, psi
        return self

    def set_signals(self, lv_left, lv_right):
        """Яркость меток физических датчиков [левый, правый]: 0 — белое под датчиком, 1 — чёрное."""
        for (halo, core), lv, color in zip(self.markers, (lv_left, lv_right), board_colors(self.st, self.sensor)):
            lv = float(np.clip(lv, 0, 1))
            for ring in halo:
                ring.set_fill(opacity=self.st.glow * lv / 2)
            core.set_fill(marker_core(color, lv))
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
