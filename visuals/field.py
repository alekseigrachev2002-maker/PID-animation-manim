"""Поле: линии трассы и разметка (жёлтые квадраты). Положение на экране задаёт камера."""
import numpy as np
from manim import VGroup, VMobject

from params import Sensor, Style

Z_FIELD = 0


class Field(VGroup):
    def __init__(self, camera, track, sensor=Sensor(), st=Style()):
        super().__init__()
        self.camera = camera
        self.drawn = camera.center.copy()     # центр камеры, при котором линии нарисованы
        for x0, y0, x1, y1 in getattr(track, "squares", []):
            m = VMobject(stroke_color=st.mark, stroke_width=camera.fr.stroke_mm(st.mark_mm))
            m.set_points_as_corners(camera.to_screen([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]))
            self.add(m)                           # под линиями: добавлены раньше
        for ln in track.lines:
            m = VMobject(stroke_color=st.line, stroke_width=camera.fr.stroke_mm(sensor.line_width))
            m.set_points_as_corners(camera.to_screen(ln.points))
            self.add(m)
        self.set_z_index(Z_FIELD)

    def update_view(self):
        """Сдвинуть линии вслед за камерой."""
        d = (self.drawn - self.camera.center) * self.camera.fr.mm
        if np.any(d):
            self.shift([d[0], d[1], 0])
            self.drawn = self.camera.center.copy()
        return self
