"""Мир под роботом: линии трассы. Робот стоит в fr.anchor, мир сдвигается (север вверх)."""
import numpy as np
from manim import VGroup, VMobject

from params import Geometry, Sensor, Style

Z_WORLD = 0
Z_XRAY = 5      # над шасси (z=4)
DASH = (6.0, 4.0)   # пунктир «рентгена»: штрих, пробел, мм


class World(VGroup):
    def __init__(self, fr, track, sensor=Sensor(), st=Style()):
        super().__init__()
        self.fr, self.track = fr, track
        self.xy = np.zeros(2)
        for ln in track.lines:
            m = VMobject(stroke_color=st.line, stroke_width=fr.stroke_mm(sensor.line_width))
            m.set_points_as_corners(fr.world(ln.points, self.xy))
            if ln.closed:
                m.close_path()
            self.add(m)
        self.set_z_index(Z_WORLD)

    def set_robot_xy(self, x, y):
        new = np.array([x, y], float)
        d = (self.xy - new) * self.fr.mm
        self.shift([d[0], d[1], 0])
        self.xy = new
        return self


def under_chassis(body_pts, geom: Geometry, margin):
    """Точки (система робота, мм), лежащие на шасси: внутри круга и вне вырезов под колёса."""
    x, y = np.abs(body_pts[:, 0]), np.abs(body_pts[:, 1])
    x0, x1 = geom.cutout_x
    in_cut = (x > x0 - margin) & (x < x1 + margin) & (y < geom.cutout_h / 2 + margin)
    return (np.hypot(x, y) < geom.chassis_r - margin) & ~in_cut


def runs(mask):
    """Отрезки подряд идущих True: [(начало, конец)), конец не включён."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


class XRay(VMobject):
    """Линия под непрозрачным шасси — пунктиром поверх корпуса. Штрихи привязаны к длине линии."""

    def __init__(self, fr, track, geom=Geometry(), sensor=Sensor(), st=Style()):
        super().__init__(stroke_color=st.xray_color, stroke_width=fr.stroke_mm(sensor.line_width),
                         stroke_opacity=st.xray_opacity)
        self.fr, self.geom, self.half_w = fr, geom, sensor.line_width / 2
        self.lines = []
        for ln in track.lines:
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ln.points, axis=0), axis=1))])
            self.lines.append((ln.points, s))
        self.set_z_index(Z_XRAY)

    def set_pose(self, x, y, psi):
        self.clear_points()
        c, s_ = np.cos(psi), np.sin(psi)
        for pts, s in self.lines:
            rel = pts - (x, y)
            body = rel @ np.array([[c, -s_], [s_, c]])
            dash_on = np.mod(s, sum(DASH)) < DASH[0]
            for a, b in runs(under_chassis(body, self.geom, self.half_w) & dash_on):
                if b - a >= 2:
                    seg = self.fr.world(pts[a:b], (x, y))
                    self.start_new_path(seg[0])
                    self.add_points_as_corners(seg[1:])
        return self
