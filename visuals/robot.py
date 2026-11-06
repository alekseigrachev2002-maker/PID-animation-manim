"""Робот сверху. Строится в мм (система робота: +y вперёд), на экран — через Frame (масштаб и точка оси)."""
import numpy as np
import math

from manim import UL, WHITE, Arc, interpolate_color, ManimColor, Circle, Difference, Rectangle, RoundedRectangle, Union, VGroup, VMobject

from params import Geometry, Style
from visuals.panels import text

HALO_R = (11.0, 8.0, 5.5)   # ореол метки: круги внахлёст, к краю свечение слабее
N_TREAD = 18          # шевронов ёлочки по окружности колеса
Z = dict(wheel=1, tread=2, board=3, chassis=4, marker=5, label=6)


def _rect(w, h, cx, cy, r=0.0, **kw):
    m = RoundedRectangle(corner_radius=r, width=w, height=h, **kw) if r else Rectangle(width=w, height=h, **kw)
    return m.move_to([cx, cy, 0])


def chassis(g: Geometry, st: Style, edge_w):
    """Шасси: круг с вырезами под колёса. Глянец — VGroup(корпус, блик), мат — VGroup(корпус)."""
    x0, x1 = g.cutout_x
    cuts = [_rect(x1 - x0, g.cutout_h, s * (x0 + x1) / 2, 0, g.cutout_r) for s in (-1, 1)]
    body = Difference(Circle(radius=g.chassis_r), Union(*cuts))
    body.set_stroke(color=st.chassis_edge, width=edge_w)
    if st.chassis_finish == "flat":
        return VGroup(body.set_fill(color=st.chassis_flat, opacity=st.chassis_opacity))
    body.set_fill(color=[st.chassis, st.chassis_hi], opacity=st.chassis_opacity).set_sheen_direction(UL)
    # блик глянца: дуга у края сверху слева; при повороте робота стоит на месте (свет неподвижен)
    glare = Arc(radius=g.chassis_r - 7, start_angle=math.radians(105), angle=math.radians(50))
    glare.set_stroke(color=st.glare, width=4 * edge_w, opacity=st.glare_opacity)
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


def markers(g: Geometry, st: Style):
    """Метки оптического центра на платах [L, R]: ореол цвета сигнала и ядро."""
    out = VGroup()
    for s, color in ((-1, st.L), (1, st.R)):
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
        self.markers = markers(geom, st)
        parts = dict(wheel=wheels(geom, st), tread=self.treads, board=boards(geom, st),
                     chassis=chassis(geom, st, fr.stroke(1.2)), marker=self.markers)
        self.body = parts["chassis"][0]
        self.glare = parts["chassis"][1] if len(parts["chassis"]) > 1 else None
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
        d = psi - self.psi
        self.rotate(d, about_point=self.fr.anchor)
        if self.glare is not None:   # свет неподвижен: блик и градиент не вращаются с роботом
            self.glare.rotate(-d, about_point=self.fr.anchor)
            self.body.set_sheen_direction(UL)
        self.psi = psi
        return self

    def set_signals(self, lvL, lvR):
        """Яркость меток датчиков: 0 — белое под датчиком, 1 — чёрное."""
        for (halo, core), lv, color in zip(self.markers, (lvL, lvR), (self.st.L, self.st.R)):
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


class PwmLabels(VGroup):
    """Числа ШИМ у левого и правого колеса. Ярлыки ездят с колёсами, текст всегда прямо.

    Число собирается из фиксированных слотов под символы: состав объектов не меняется от кадра к кадру,
    иначе Manim рисует устаревшие копии (список объектов он собирает в начале анимации).
    """
    SLOTS = 4   # знак и три цифры

    def __init__(self, fr, st=Style(), x_mm=108.0):
        super().__init__()
        self.fr, self.st, self.x_mm = fr, st, x_mm
        self.cache = {}
        self.pills = VGroup(*[RoundedRectangle(corner_radius=6 * fr.unit, width=56 * fr.unit, height=28 * fr.unit,
                                               stroke_width=0, fill_color=st.block, fill_opacity=0.92)
                              for _ in range(2)])
        self.slots = VGroup(*[VGroup(*[VMobject(fill_color=st.text, fill_opacity=1, stroke_width=0)
                                       for _ in range(self.SLOTS)]) for _ in range(2)])
        self.add(self.pills, self.slots)
        self.set_z_index(Z["label"])
        self.slots.set_z_index(Z["label"] + 1)
        self.set_values(0.0, 0.0, 0.0)

    def _glyphs(self, v):
        """Контуры символов числа v относительно его центра (кэш по значению)."""
        k = int(round(v))
        if k not in self.cache:
            t = text(self.fr, str(k).replace("-", "−"), 16, self.st.text, self.st)
            c = t.get_center()
            self.cache[k] = [g.get_points() - c for g in t]
        return self.cache[k]

    def set_values(self, psi, pwm_l, pwm_r):
        c, s = np.cos(psi), np.sin(psi)
        for i, (side, v) in enumerate(((-1, pwm_l), (1, pwm_r))):
            x = side * self.x_mm
            pos = self.fr.anchor + np.array([c * x, s * x, 0]) * self.fr.mm
            self.pills[i].move_to(pos)
            glyphs = self._glyphs(v)
            for j, slot in enumerate(self.slots[i]):
                if j < len(glyphs):
                    slot.set_points(glyphs[j] + pos)
                else:
                    slot.clear_points()
        return self
