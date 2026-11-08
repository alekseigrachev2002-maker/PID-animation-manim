"""Панель графиков (левые 2/3 кадра): шапка, блок формулы, график, легенда, доп. легенда.

    panel = GraphPanel(fr, st, "Подбираем P", "kp = 0.3, едет по линии",
                       y=("err", -1023, 1023), t_max=2.2, t_ticks=[0, 0.5, 1, 1.5, 2],
                       signals=[ERR], readout=dict(signs=(1, -1), law="kp·err"))
    self.play(panel.set_formula("control = kp·err"))
    curves = panel.curves(traj, self.t)            # кривые всех сигналов, растут за трекером t

Сигнал — что рисуем на графике: подпись в легенде, поле траектории, цвет.
Доп. легенда — значения без кривой на графике (ШИМ колёс); обновляется каждый кадр через update_state.
"""
import warnings
from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, DashedLine, Dot, FadeIn, FadeOut, Line, Rectangle, RoundedRectangle,
                   Succession, ValueTracker, VGroup, VMobject)

from params import Style
from visuals.layout import Z_UI, baseline, on_baseline, pick_font, text

ORIGIN_PT = np.zeros(3)


@dataclass(frozen=True)
class Signal:
    label: str           # подпись в легенде
    key: str             # поле Trajectory: "L", "R", "e", "uP", "uD"
    color: str           # имя цвета в Style ("L", "R", "e", "D") или "#RRGGBB"
    width_px: float = 3.0     # все кривые одной ширины


ERR = Signal("err = L − R", "e", "e")


def color_of(st, c):
    return c if c.startswith("#") else getattr(st, c)


def swap(old, new, shift=0.08):
    """Смена текста: старый гаснет, уходя вверх, затем новый проявляется снизу (без морфинга букв)."""
    return Succession(FadeOut(old, shift=shift * UP), FadeIn(new, shift=shift * UP))


# ---------- шапка ----------

class Header(VGroup):
    """«Фраза этапа: пояснение шага» слева, блок с формулой у правого края панели."""

    def __init__(self, fr, phrase, caption, st=Style()):
        super().__init__()
        self.fr, self.st = fr, st
        self.phrase = self._phrase(phrase)
        self.caption_text = caption
        self.caption = self._caption(caption)
        self.block = None
        self.add(self.phrase, self.caption)

    def _phrase(self, s):
        t = text(self.fr, s + ":", 22, self.st.text, self.st, weight="SEMIBOLD")
        return t.move_to(self.fr.px(self.fr.layout.margin_px, self.fr.layout.header_y_px),
                         aligned_edge=LEFT).set_z_index(Z_UI)

    def _caption(self, s):
        t = text(self.fr, s, 22, self.st.muted, self.st).next_to(self.phrase, RIGHT, buff=10 * self.fr.unit)
        return on_baseline(t, baseline(self.phrase)).set_z_index(Z_UI)

    def _block(self, s, colors):
        """Формула как фрагмент кода: моноширинный шрифт на тёмной подложке, части формулы — цветом своих кривых.

        colors: [(часть формулы, цвет)] — все вхождения, или (часть, цвет, n) — только n-е вхождение (с нуля);
        например [("err", st.e)] или [("-150", L), ("150", R, 1)]. Позже в списке — главнее.
        """
        fr, st = self.fr, self.st
        t = text(fr, s, 19, st.text, st, font=pick_font(st.code_font))
        glyphs = s.replace(" ", "")
        for part, color, *n in colors:
            part = part.replace(" ", "")
            hits, i = [], glyphs.find(part)
            while i >= 0:
                hits.append(i)
                i = glyphs.find(part, i + 1)
            for i in (hits if not n else hits[n[0]:n[0] + 1]):
                for k in range(i, i + len(part)):
                    t[k].set_color(color)
        t.move_to(fr.px(fr.layout.graph_w_px - fr.layout.margin_px - 14, 0), aligned_edge=RIGHT)
        on_baseline(t, baseline(self.phrase))
        box = RoundedRectangle(corner_radius=6 * fr.unit, width=t.width + 28 * fr.unit, height=36 * fr.unit,
                               stroke_width=0, fill_color=st.block, fill_opacity=1).move_to(t)
        return VGroup(box, t).set_z_index(Z_UI)

    def _check_fit(self):
        """Пояснение не должно налезать на блок формулы — иначе предупреждение при рендере."""
        if self.block is not None and self.caption.get_right()[0] > self.block.get_left()[0] - 12 * self.fr.unit:
            warnings.warn(f"шапка: пояснение «{self.caption_text}» налезает на блок формулы — сократите текст")

    def _swap(self, attr, new):
        old = getattr(self, attr)
        setattr(self, attr, new)
        if old is None:
            return FadeIn(new, shift=0.1 * DOWN)
        self.remove(old)
        return swap(old, new)

    def set_formula(self, s, colors=()):
        anim = self._swap("block", self._block(s, colors))
        self._check_fit()
        return anim

    def set_caption(self, s):
        self.caption_text = s
        anim = self._swap("caption", self._caption(s))
        self._check_fit()
        return anim

    def set_phrase(self, phrase, caption):
        """Смена этапа: новая фраза и пояснение."""
        return [self._swap("phrase", self._phrase(phrase)), self.set_caption(caption)]


# ---------- легенда ----------

def legend_baseline(fr):
    return fr.px(0, fr.layout.legend_y_px + 5)[1]


def legend(fr, items, x_px=None, st=Style()):
    """Одна строка внизу панели. items: Signal или (Signal, непрозрачность)."""
    x0 = fr.layout.plot_px[0] if x_px is None else x_px
    y = legend_baseline(fr)
    g, x = VGroup(), fr.px(x0, 0)[0]
    for it in items:
        sig, op = it if isinstance(it, tuple) else (it, 1.0)
        lab = text(fr, sig.label, 16, st.text, st)
        sw = Line(ORIGIN_PT, RIGHT * 24 * fr.unit, color=color_of(st, sig.color), stroke_width=fr.stroke(3),
                  stroke_opacity=op)
        sw.move_to([x, y + 5 * fr.unit, 0], aligned_edge=LEFT)          # середина строчных букв
        lab.move_to([sw.get_right()[0] + 10 * fr.unit, 0, 0], aligned_edge=LEFT)
        g.add(VGroup(sw, on_baseline(lab, y)))
        x = lab.get_right()[0] + 36 * fr.unit
    return g.set_z_index(Z_UI)


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

    def _zero(self):
        return float(np.clip(0.0, self.lo.get_value(), self.hi.get_value()))

    def axes(self, ticks, y_label, t_ticks):
        """Оси с подписями. ticks: {значение: функция непрозрачности от (lo, hi)}."""
        fr, st = self.fr, self.st
        x0, y0, x1, y1 = fr.layout.plot_px
        w = fr.stroke(1.2)
        y_axis = Line(fr.px(x0, y0), fr.px(x0, y1), color=st.axis, stroke_width=w)
        x_axis = Line(fr.px(x0, y1), fr.px(x1, y1), color=st.axis, stroke_width=w)
        x_axis.add_updater(lambda m: m.put_start_and_end_on(*self.to_screen([0, self.t_max], [self._zero()] * 2)))
        g = VGroup(y_axis, x_axis)
        for tk in t_ticks:
            g.add(text(fr, f"{tk:g}", 13, st.muted, st).move_to(fr.px(x0 + tk / self.t_max * (x1 - x0), y1 + 16)))
        g.add(text(fr, "t, с", 13, st.muted, st).move_to(fr.px(x1 + 14, y1 + 16), aligned_edge=LEFT))
        g.add(text(fr, y_label, 13, st.muted, st).move_to(fr.px(x0 + 8, y0 - 12), aligned_edge=LEFT))
        for val, op in ticks.items():
            lab = text(fr, f"{val:g}".replace("-", "−"), 13, st.muted, st)
            lab.add_updater(lambda m, val=val, op=op: m.move_to(
                self.to_screen([0], [val])[0] + LEFT * 10 * fr.unit, aligned_edge=RIGHT).set_opacity(
                op(self.lo.get_value(), self.hi.get_value())))
            g.add(lab)
        return g.set_z_index(Z_UI)

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

    def curve(self, tt, vals, color, t_end, opacity=None, width_px=3.0):
        """Растущая кривая до момента t_end (ValueTracker или функция) с точкой на конце."""
        fr = self.fr
        get_end = t_end.get_value if hasattr(t_end, "get_value") else t_end
        line = VMobject(stroke_color=color, stroke_width=fr.stroke(width_px))
        dot = Dot(radius=4 * fr.unit, color=color)

        def upd(m):
            te = get_end()
            i = int(np.searchsorted(tt, te, side="right"))
            ts = np.append(tt[:i], te) if i < len(tt) else tt
            pts = self.to_screen(ts, np.interp(ts, tt, vals))
            if len(pts) < 2:
                pts = np.vstack([pts, pts])
            m[0].set_points_as_corners(pts)
            m[1].move_to(pts[-1])
            op = 1.0 if opacity is None else opacity.get_value()
            m[0].set_stroke(opacity=op)
            m[1].set_opacity(op if te > tt[0] else 0)

        g = VGroup(line, dot).add_updater(upd)
        upd(g)
        return g.set_z_index(Z_UI + 1)


# ---------- доп. легенда: ШИМ колёс ----------

class Number(VGroup):
    """Целое число из фиксированных слотов под символы; левый край и линия шрифта задаются точкой.

    Состав объектов не меняется от кадра к кадру: Manim собирает список объектов в начале анимации,
    и подмена текстовых объектов на лету оставляла бы устаревшие копии.
    """
    SLOTS = 4   # знак и три цифры

    def __init__(self, fr, px, color, st=Style()):
        super().__init__(*[VMobject(fill_color=color, fill_opacity=1, stroke_width=0) for _ in range(self.SLOTS)])
        self.fr, self.px, self.color, self.st = fr, px, color, st
        self.cache = {}

    def _glyphs(self, k):
        if k not in self.cache:
            t = text(self.fr, str(k).replace("-", "−"), self.px, self.color, self.st)
            ref = np.array([t.get_left()[0], t[-1].get_bottom()[1], 0.0])
            self.cache[k] = [g.get_points() - ref for g in t]
        return self.cache[k]

    def set_value(self, v, left_base):
        glyphs = self._glyphs(int(round(v)))
        for j, slot in enumerate(self):
            if j < len(glyphs):
                slot.set_points(glyphs[j] + left_base)
            else:
                slot.clear_points()
        return self


class MotorReadout(VGroup):
    """«▬ левый мотор 150 + kp·err = 255   ▬ правый мотор 150 − kp·err = 45» — ШИМ физических колёс.

    Полоска — цвет оси колеса на роботе; знаки — из подключения моторов (model.wheel_signs).
    Число стоит сразу после «=»; следующая группа отстоит на ширину самого длинного числа — строка не дёргается.
    opacity — ValueTracker: приглушать на время возврата между прогонами.
    """
    NUM_W = 40   # место под самое длинное число («−255»), px
    PX = 15      # кегль строки, px

    def __init__(self, fr, x_px, signs, law="kp·err", st=Style()):
        super().__init__()
        self.fr, self.st = fr, st
        self.opacity = ValueTracker(1.0)
        y = legend_baseline(fr)
        self.texts, self.numbers, self.anchors = VGroup(), VGroup(), []
        x = fr.px(x_px, 0)[0]
        for name, sign, color in (("левый мотор", signs[0], st.L), ("правый мотор", signs[1], st.R)):
            sw = Rectangle(width=14 * fr.unit, height=7 * fr.unit, stroke_width=0, fill_color=color, fill_opacity=1)
            sw.move_to([x, y + 5 * fr.unit, 0], aligned_edge=LEFT)
            lab = text(fr, name, self.PX, st.text, st)
            lab.move_to([sw.get_right()[0] + 8 * fr.unit, 0, 0], aligned_edge=LEFT)
            frm = text(fr, f"150 {'+' if sign > 0 else '−'} {law} =", self.PX, st.muted, st)
            on_baseline(lab, y)
            frm.move_to([lab.get_right()[0] + 10 * fr.unit, 0, 0], aligned_edge=LEFT)
            on_baseline(frm, y)
            left = frm.get_right()[0] + 6 * fr.unit
            self.anchors.append(np.array([left, y, 0.0]))
            self.texts.add(VGroup(sw, lab, frm))
            self.numbers.add(Number(fr, self.PX, st.text, st))
            x = left + (self.NUM_W + 32) * fr.unit
        self.add(self.texts, self.numbers)
        self.set_z_index(Z_UI)
        self.update_state({"pwmL": 0.0, "pwmR": 0.0})

    def update_state(self, state):
        for num, anchor, v in zip(self.numbers, self.anchors, (state["pwmL"], state["pwmR"])):
            num.set_value(v, anchor)
        op = self.opacity.get_value()
        self.texts.set_opacity(op)
        self.numbers.set_fill(opacity=op)
        return self


# ---------- панель целиком ----------

class GraphPanel(VGroup):
    """Шапка + блок формулы + график с сигналами + легенда из сигналов + доп. легенда (ШИМ колёс)."""

    def __init__(self, fr, st, phrase, caption, y, t_max, t_ticks, signals, ticks=None, readout=None):
        super().__init__()
        self.fr, self.st, self.signals = fr, st, list(signals)
        y_label, lo, hi = y
        self.header = Header(fr, phrase, caption, st)
        self.plot = Plot(fr, t_max, lo, hi, st)
        ticks = ticks or {v: (lambda lo_, hi_: 1) for v in sorted({lo, 0, hi})}
        self.axes = self.plot.axes(ticks, y_label, t_ticks)
        self.legend = legend(fr, self.signals, st=st)
        self.add(self.header, self.axes, self.legend)
        self.readout = None
        if readout is not None:
            x_px = fr.to_px_x(self.legend.get_right()[0]) + 40
            self.readout = MotorReadout(fr, x_px, st=st, **readout)

    def set_formula(self, s, colors=(("err", "e"),)):
        """Формула в шапке; colors — какие части каким цветом: [("err", "e"), ("kd·Δerr", "D")]."""
        return self.header.set_formula(s, [(part, color_of(self.st, c), *n) for part, c, *n in colors])

    def set_caption(self, s):
        return self.header.set_caption(s)

    def set_legend(self, items):
        """Сменить легенду (анимация). items: Signal или (Signal, непрозрачность)."""
        new = legend(self.fr, items, st=self.st)
        old, self.legend = self.legend, new
        self.remove(old)
        return swap(old, new)

    def curve(self, traj, sig, t_end, opacity=None):
        return self.plot.curve(traj.tt, getattr(traj, sig.key), color_of(self.st, sig.color), t_end, opacity,
                               sig.width_px)

    def curves(self, traj, t_end, opacity=None):
        """Кривые всех сигналов панели для траектории traj."""
        return VGroup(*[self.curve(traj, s, t_end, opacity) for s in self.signals])
