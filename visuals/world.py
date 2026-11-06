"""Мир под роботом: линии трассы. Робот стоит в fr.anchor, мир сдвигается (север вверх)."""
import numpy as np
from manim import VGroup, VMobject

from params import Sensor, Style

Z_WORLD = 0


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
