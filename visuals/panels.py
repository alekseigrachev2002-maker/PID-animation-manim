"""Макет кадра: фон, шапка, легенда, график. Перевод мм и пикселей в единицы Manim — только здесь (Frame)."""
import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, DashedLine, Dot, Line, MarkupText, Rectangle, RoundedRectangle,
                   Text, ValueTracker, VGroup, VMobject, config, FadeIn, FadeTransform)

from params import Layout, Style

Z_PANEL = 10   # панель графиков закрывает мир
Z_UI = 12      # всё, что на панели


class Frame:
    """Пиксели эталонного кадра 1280×720 и миллиметры → единицы Manim."""

    def __init__(self, layout=Layout()):
        self.layout = layout
        self.unit = config.frame_width / layout.width_px            # единиц Manim на пиксель
        self.height_px = layout.width_px * config.frame_height / config.frame_width
        self.mm = layout.px_per_mm * self.unit                       # единиц Manim на мм

    def px(self, x, y):
        return np.array([(x - self.layout.width_px / 2) * self.unit, (self.height_px / 2 - y) * self.unit, 0.0])

    def font(self, px):
        """Кегль Manim для шрифта высотой px пикселей (em)."""
        return px * self.unit * 72

    def stroke(self, px):
        """stroke_width Manim для линии толщиной px пикселей."""
        return px * self.unit / 0.01

    def stroke_mm(self, mm):
        return mm * self.mm / 0.01

    @property
    def anchor(self):
        """Экранная точка середины оси колёс."""
        return self.px(*self.layout.robot_px)

    def world(self, pts_mm, robot_xy):
        """Точки мира (мм, (N,2)) → экран при роботе в robot_xy; север вверх, робот в anchor."""
        p = (np.asarray(pts_mm, float) - np.asarray(robot_xy, float)) * self.mm
        return np.column_stack([p, np.zeros(len(p))]) + self.anchor


def text(fr, s, px, color, st=Style(), **kw):
    return Text(s, font=st.font, font_size=fr.font(px), color=color, **kw)


def backdrop(fr, st=Style()):
    field = Rectangle(width=config.frame_width, height=config.frame_height, fill_color=st.field,
                      fill_opacity=1, stroke_width=0).set_z_index(-10)
    w = fr.layout.graph_w_px * fr.unit
    panel = Rectangle(width=w, height=config.frame_height, fill_color=st.graph_bg, fill_opacity=1,
                      stroke_width=0).move_to(fr.px(fr.layout.graph_w_px / 2, fr.height_px / 2))
    return VGroup(field, panel.set_z_index(Z_PANEL))


# ---------- шапка ----------

class Header(VGroup):
    """Шапка: «Фраза этапа: пояснение шага» слева и блок с текущей формулой у правого края панели.

    Фраза меняется только на смене этапа, пояснение (4–6 слов) — на каждом шаге сцены.
    """

    def __init__(self, fr, phrase, caption, st=Style()):
        super().__init__()
        self.fr, self.st = fr, st
        self.phrase = self._phrase(phrase)
        self.caption = self._caption(caption)
        self.block = None
        self.add(self.phrase, self.caption)

    def _phrase(self, s):
        m = self.fr.layout.margin_px
        t = text(self.fr, s + ":", 22, self.st.text, self.st, weight="SEMIBOLD")
        return t.move_to(self.fr.px(m, self.fr.layout.header_y_px), aligned_edge=LEFT).set_z_index(Z_UI)

    def _caption(self, s):
        t = text(self.fr, s, 22, self.st.muted, self.st)
        t.next_to(self.phrase, RIGHT, buff=10 * self.fr.unit)
        return t.align_to(self.phrase, DOWN).set_z_index(Z_UI)

    def _block(self, markup, color):
        fr, st = self.fr, self.st
        t = MarkupText(markup, font=st.font, font_size=fr.font(20), color=st.text)
        box = RoundedRectangle(corner_radius=8 * fr.unit, width=t.width + 40 * fr.unit, height=44 * fr.unit,
                               stroke_color=color, stroke_width=fr.stroke(2), fill_color=color, fill_opacity=0.14)
        g = VGroup(box, t.move_to(box))
        right = fr.layout.graph_w_px - fr.layout.margin_px
        g.move_to(fr.px(right, fr.layout.header_y_px), aligned_edge=RIGHT)
        return g.set_z_index(Z_UI)

    def _swap(self, attr, new):
        old = getattr(self, attr)
        setattr(self, attr, new)
        self.add(new)
        if old is None:
            return FadeIn(new, shift=0.1 * DOWN)
        self.remove(old)
        return FadeTransform(old, new)

    def set_formula(self, markup, color):
        return self._swap("block", self._block(markup, color))

    def set_caption(self, s):
        return self._swap("caption", self._caption(s))

    def set_phrase(self, phrase, caption):
        """Смена этапа: новая фраза и пояснение."""
        anim = self._swap("phrase", self._phrase(phrase))
        return [anim, self.set_caption(caption)]


# ---------- легенда ----------

def legend(fr, items, st=Style()):
    """items: [(цвет, подпись, непрозрачность)] — одна строка внизу панели графиков."""
    x0, y = fr.layout.plot_px[0], fr.layout.legend_y_px
    g = VGroup()
    for color, label, op in items:
        sw = Line(ORIGIN_PT, RIGHT * 24 * fr.unit, color=color, stroke_width=fr.stroke(3), stroke_opacity=op)
        lab = text(fr, label, 16, st.text, st).next_to(sw, RIGHT, buff=10 * fr.unit)
        g.add(VGroup(sw, lab))
    g.arrange(RIGHT, buff=36 * fr.unit)
    g.move_to(fr.px(x0, y), aligned_edge=LEFT)
    return g.set_z_index(Z_UI)


ORIGIN_PT = np.zeros(3)


# ---------- график ----------

class Plot:
    """Оси t–значение в области plot_px; диапазон по вертикали — трекеры lo, hi (можно анимировать)."""

    def __init__(self, fr, t_max, lo, hi, st=Style()):
        self.fr, self.st, self.t_max = fr, st, t_max
        self.lo, self.hi = ValueTracker(lo), ValueTracker(hi)

    def to_screen(self, t, v):
        x0, y0, x1, y1 = self.fr.layout.plot_px
        lo, hi = self.lo.get_value(), self.hi.get_value()
        X = x0 + (np.asarray(t, float) / self.t_max) * (x1 - x0)
        Y = y1 - (np.asarray(v, float) - lo) / (hi - lo) * (y1 - y0)
        u = self.fr.unit
        return np.column_stack([(X - self.fr.layout.width_px / 2) * u, (self.fr.height_px / 2 - Y) * u,
                                np.zeros(np.size(X))])

    def axes(self, ticks, y_label="АЦП", t_ticks=None):
        """Оси с подписями. ticks: {значение: функция непрозрачности от (lo, hi)}."""
        fr, st = self.fr, self.st
        x0, y0, x1, y1 = fr.layout.plot_px
        t_ticks = range(int(self.t_max) + 1) if t_ticks is None else t_ticks
        w = fr.stroke(1.2)
        y_axis = Line(fr.px(x0, y0), fr.px(x0, y1), color=st.axis, stroke_width=w)
        x_axis = Line(fr.px(x0, y1), fr.px(x1, y1), color=st.axis, stroke_width=w)
        x_axis.add_updater(lambda m: m.put_start_and_end_on(*self.to_screen([0, self.t_max], [self._zero()] * 2)))
        g = VGroup(y_axis, x_axis)
        for tk in t_ticks:
            g.add(text(fr, str(tk), 13, st.muted, st).move_to(fr.px(x0 + tk / self.t_max * (x1 - x0), y1 + 16)))
        g.add(text(fr, "t, с", 13, st.muted, st).move_to(fr.px(x1, y1 + 16), aligned_edge=RIGHT))
        g.add(text(fr, y_label, 13, st.muted, st).move_to(fr.px(x0 + 8, y0 - 12), aligned_edge=LEFT))
        for val, op in ticks.items():
            lab = text(fr, f"{val:g}".replace("-", "−"), 13, st.muted, st)
            lab.add_updater(lambda m, val=val, op=op: m.move_to(
                self.to_screen([0], [val])[0] + LEFT * 10 * fr.unit, aligned_edge=RIGHT).set_opacity(
                op(self.lo.get_value(), self.hi.get_value())))
            g.add(lab)
        return g.set_z_index(Z_UI)

    def _zero(self):
        return float(np.clip(0.0, self.lo.get_value(), self.hi.get_value()))

    def hline(self, value, label, opacity):
        """Пунктир-ориентир с подписью справа; opacity — ValueTracker."""
        fr, st = self.fr, self.st
        ln = DashedLine(ORIGIN_PT, RIGHT, color=st.muted, stroke_width=fr.stroke(1), dash_length=4 * fr.unit)
        lab = text(fr, label, 12, st.muted, st)

        def upd(m):
            a, b = self.to_screen([0, self.t_max], [value, value])
            m[0].put_start_and_end_on(a, b)
            m[1].next_to(b, UP, buff=4 * fr.unit).align_to(b, RIGHT)
            m.set_opacity(opacity.get_value())

        g = VGroup(ln, lab).add_updater(upd)
        upd(g)
        return g.set_z_index(Z_UI)

    def curve(self, tt, vals, color, t_end, opacity=None, width_px=2.5, head=True):
        """Растущая кривая до момента t_end (ValueTracker или функция) с точкой на конце."""
        fr = self.fr
        get_end = t_end.get_value if hasattr(t_end, "get_value") else t_end
        line = VMobject(stroke_color=color, stroke_width=fr.stroke(width_px))
        dot = Dot(radius=4 * fr.unit, color=color)

        def upd(m):
            te = get_end()
            i = int(np.searchsorted(tt, te, side="right"))
            ts = np.append(tt[:i], te) if i < len(tt) else tt
            vs = np.interp(ts, tt, vals)
            pts = self.to_screen(ts, vs)
            if len(pts) < 2:
                pts = np.vstack([pts, pts])
            m[0].set_points_as_corners(pts)
            m[1].move_to(pts[-1])
            op = 1.0 if opacity is None else opacity.get_value()
            m[0].set_stroke(opacity=op)
            m[1].set_opacity(op if head and te > tt[0] else 0)

        g = VGroup(line, dot).add_updater(upd)
        upd(g)
        return g.set_z_index(Z_UI + 1)
