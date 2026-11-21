"""Поле: линии трассы, разметка (желтые квадраты), черные круги и кубики. Положение и поворот на экране задает
камера (Camera.rot — поворот на всю сцену)."""
import numpy as np
from manim import Circle, Polygon, VGroup, VMobject

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
        for c, r in getattr(track, "discs", []):              # черные круги — на маршрут не попадают
            self.add(Circle(radius=r * camera.fr.mm, stroke_width=0, fill_color=st.line, fill_opacity=1)
                     .move_to(camera.point(*c)))
        for b in getattr(track, "blocks", []):                # кубики — поверх линий
            h, (c, s) = b.side / 2, (np.cos(b.angle), np.sin(b.angle))
            loc = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ np.array([[c, s], [-s, c]])
            self.add(Polygon(*camera.to_screen(loc + np.asarray(b.center, float)), fill_color=b.color,
                             fill_opacity=1, stroke_color=b.edge, stroke_width=camera.fr.stroke(2.0)))
        self.set_z_index(Z_FIELD)

    def update_view(self):
        """Сдвинуть поле вслед за камерой (с ее поворотом)."""
        d = self.camera.R @ (self.drawn - self.camera.center) * self.camera.fr.mm
        if np.any(d):
            self.shift([d[0], d[1], 0])
            self.drawn = self.camera.center.copy()
        return self
