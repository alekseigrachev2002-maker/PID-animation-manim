"""Панель графиков (левые 2/3 кадра): шапка, блок формулы, график, легенда, доп. легенда.

    panel = GraphPanel(fr, st, "Подбираем P", "kp = 0.3, едет по линии",
                       y=("err", -1023, 1023), t_max=2.2, t_ticks=[0, 0.5, 1, 1.5, 2],
                       signals=[ERR], readout=dict(signs=(1, -1)))
    self.play(panel.set_formula("control = err * kp"))
    curves = panel.curves(traj, self.t)            # кривые всех сигналов, растут за трекером t

Сигнал — что рисуем на графике: подпись в легенде, поле траектории, цвет.
Доп. легенда — значения без кривой на графике (ШИМ колес); обновляется каждый кадр через update_state.
"""
import warnings
from dataclasses import dataclass

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, DashedLine, Dot, FadeIn, FadeOut, Line, Rectangle, RoundedRectangle,
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
    """Заголовок этапа (кегль заголовка слайда); под ним — блок с формулой и сразу за ним пояснение шага."""

    def __init__(self, fr, phrase, caption, st=Style()):
        super().__init__()
        self.fr, self.st = fr, st
        self.block = None
        self.phrase = self._phrase(phrase)
        self.caption_text = caption
        self.caption = self._caption(caption)
        self.add(self.phrase, self.caption)

    def _phrase(self, s):
        t = text(self.fr, s, self.st.title_px, self.st.text, self.st, weight=self.st.title_weight)
        return t.move_to(self.fr.px(self.fr.layout.margin_px, self.fr.layout.header_y_px),
                         aligned_edge=LEFT).set_z_index(Z_UI)

    def _caption_x(self):
        """Левый край пояснения: за блоком формулы, без блока — у края панели."""
        fr = self.fr
        return fr.px(fr.layout.margin_px, 0)[0] if self.block is None else self.block.get_right()[0] + 16 * fr.unit

    def _caption(self, s):
        fr = self.fr
        t = text(fr, s, self.st.text_px, self.st.muted, self.st)
        t.move_to([self._caption_x(), 0, 0], aligned_edge=LEFT)
        return on_baseline(t, row_baseline(fr, fr.layout.caption_y_px, self.st)).set_z_index(Z_UI)

    def _block(self, s, colors):
        """Формула как фрагмент кода: моноширинный шрифт на темной подложке, части формулы — цветом своих кривых.

        colors: [(часть формулы, цвет)] — все вхождения, или (часть, цвет, n) — только n-е вхождение (с нуля);
        например [("err", st.e)] или [("-150", L), ("150", R, 1)]. Позже в списке — главнее.
        """
        fr, st = self.fr, self.st
        t = text(fr, s, st.text_px, st.text, st, font=pick_font(st.code_font))
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
        t.move_to(fr.px(fr.layout.margin_px + 14, 0), aligned_edge=LEFT)
        on_baseline(t, row_baseline(fr, fr.layout.caption_y_px, st))
        box = RoundedRectangle(corner_radius=7 * fr.unit, width=t.width + 28 * fr.unit, height=44 * fr.unit,
                               stroke_width=0, fill_color=st.block, fill_opacity=1)
        box.move_to([t.get_center()[0], row_baseline(fr, fr.layout.caption_y_px, st) + 0.3 * st.text_px * fr.unit, 0])
        return VGroup(box, t).set_z_index(Z_UI)

    def _check_fit(self):
        """Заголовок не должен налезать на блок формулы, пояснение — выходить за панель; иначе предупреждение."""
        fr = self.fr
        if self.caption.get_right()[0] > fr.px(fr.layout.graph_w_px - fr.layout.margin_px, 0)[0]:
            warnings.warn(f"шапка: пояснение «{self.caption_text}» не помещается в панель — сократите текст")

    def _swap(self, attr, new):
        old = getattr(self, attr)
        setattr(self, attr, new)
        if old is None:
            return FadeIn(new, shift=0.1 * DOWN)
        self.remove(old)
        return swap(old, new)

    def set_formula(self, s, colors=()):
        """Новая формула; пояснение отъезжает, чтобы стоять сразу за блоком."""
        anim = self._swap("block", self._block(s, colors))
        dx = self._caption_x() - self.caption.get_left()[0]
        self._check_fit()
        if abs(dx) < 1e-6:
            return anim
        return AnimationGroup(anim, self.caption.animate.shift(dx * RIGHT))

    def set_caption(self, s):
        self.caption_text = s
        anim = self._swap("caption", self._caption(s))
        self._check_fit()
        return anim

    def set_phrase(self, phrase, caption):
        """Смена этапа: новая фраза и пояснение."""
        return [self._swap("phrase", self._phrase(phrase)), self.set_caption(caption)]


# ---------- легенда ----------

def row_baseline(fr, y_px, st=Style()):
    """Линия шрифта строки текста кегля text_px с серединой строчных букв на y_px."""
    return fr.px(0, y_px + 0.25 * st.text_px)[1]


def legend_baseline(fr, st=Style()):
    return row_baseline(fr, fr.layout.legend_y_px, st)


def legend(fr, items, x_px=None, st=Style()):
    """Одна строка внизу панели. items: Signal или (Signal, непрозрачность)."""
    x0 = fr.layout.margin_px if x_px is None else x_px
    y = legend_baseline(fr, st)
    g, x = VGroup(), fr.px(x0, 0)[0]
    for it in items:
        sig, op = it if isinstance(it, tuple) else (it, 1.0)
        lab = text(fr, sig.label, st.text_px, st.text, st)
        sw = Line(ORIGIN_PT, RIGHT * 30 * fr.unit, color=color_of(st, sig.color), stroke_width=fr.stroke(4),
                  stroke_opacity=op)
        sw.move_to([x, y + 0.25 * st.text_px * fr.unit, 0], aligned_edge=LEFT)          # середина строчных букв
        lab.move_to([sw.get_right()[0] + 10 * fr.unit, 0, 0], aligned_edge=LEFT)
        g.add(VGroup(sw, on_baseline(lab, y)))
        x = lab.get_right()[0] + 40 * fr.unit
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
        small = st.small_px
        for tk in t_ticks:
            g.add(text(fr, f"{tk:g}", small, st.muted, st).move_to(fr.px(x0 + tk / self.t_max * (x1 - x0), y1 + 20)))
        g.add(text(fr, "t, с", small, st.muted, st).move_to(fr.px(x1 + 14, y1 + 20), aligned_edge=LEFT))
        g.add(text(fr, y_label, small, st.muted, st).move_to(fr.px(x0 + 8, y0 - 16), aligned_edge=LEFT))
        for val, op in ticks.items():
            lab = text(fr, f"{val:g}".replace("-", "−"), small, st.muted, st)
            def place(m, val=val, op=op):
                m.move_to(self.to_screen([0], [val])[0] + LEFT * 10 * fr.unit, aligned_edge=RIGHT)
                m.set_opacity(op(self.lo.get_value(), self.hi.get_value()))

            lab.add_updater(place)
            place(lab)      # сразу на место: неподвижный wait() апдейтеры не вызывает — иначе первый кадр с подписями в центре
            g.add(lab)
        return g.set_z_index(Z_UI)

    def hline(self, value, label, opacity):
        """Пунктир-ориентир с подписью справа; opacity — ValueTracker."""
        fr, st = self.fr, self.st
        ln = DashedLine(ORIGIN_PT, RIGHT, color=st.muted, stroke_width=fr.stroke(1), dash_length=4 * fr.unit)
        lab = text(fr, label, st.small_px, st.muted, st)

        def upd(m):
            a, b = self.to_screen([0, self.t_max], [value, value])
            m[0].put_start_and_end_on(a, b)
            m[1].next_to(a, UP, buff=4 * fr.unit).align_to(a, LEFT).shift(12 * fr.unit * RIGHT)   # слева: справа идут кривые
            m.set_opacity(opacity.get_value())

        g = VGroup(ln, lab).add_updater(upd)
        upd(g)
        return g.set_z_index(Z_UI)

    def curve(self, tt, vals, color, t_end, opacity=None, width_px=3.0, peers=None):
        """Растущая кривая до момента t_end (ValueTracker или функция) с точкой на конце.

        peers — общий список кривых одной траектории: где кривые ближе друг к другу, чем толщина линии,
        они расходятся (см. SPREAD), чтобы совпадающая кривая не прятала другую.
        """
        fr = self.fr
        get_end = t_end.get_value if hasattr(t_end, "get_value") else t_end
        line = VMobject(stroke_color=color, stroke_width=fr.stroke(width_px))
        dot = Dot(radius=4 * fr.unit, color=color)
        me = dict(tt=np.asarray(tt, float), vals=np.asarray(vals, float), width=width_px, order=0)
        if peers is not None:
            me["order"] = len(peers)
            peers.append(me)

        def upd(m):
            te = get_end()
            i = int(np.searchsorted(tt, te, side="right"))
            ts = np.append(tt[:i], te) if i < len(tt) else tt
            pts = self.to_screen(ts, np.interp(ts, tt, vals))
            if peers is not None:
                pts += self._spread(me, peers, ts, pts[:, 1])[:, None] * fr.unit * self._normals(pts)
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

    SPREAD = 0.35      # сдвиг совпадающих кривых, доля толщины линии в каждую сторону

    @staticmethod
    def _normals(pts):
        """Единичные нормали к ломаной, «вверх» от нее (влево по ходу времени): на пологих участках сдвиг
        вверх-вниз, на крутых — вбок, иначе совпадающие кривые на крутом спуске не разошлись бы."""
        if len(pts) < 2:
            return np.array([[0.0, 1.0, 0.0]] * len(pts))
        tx, ty = np.gradient(pts[:, 0]), np.gradient(pts[:, 1])
        n = np.hypot(tx, ty)
        ok = n > 1e-12
        out = np.zeros_like(pts)
        out[:, 1] = 1.0
        out[ok, 0], out[ok, 1] = -ty[ok] / n[ok], tx[ok] / n[ok]
        return out

    def _spread(self, me, peers, ts, y):
        """Сдвиг кривой me «вверх» (px эталонного кадра, вдоль нормали) в точках ts; y — ее экранные координаты.

        Сдвиг полный, когда кривые совпадают, и плавно сходит на нет к расстоянию в одну толщину линии.
        Дорожки постоянные: кривая раньше в списке всегда уходит вверх, позже — вниз. Так на пересечении
        кривые просто проходят друг сквозь друга — без скачка, который давал выбор «кто выше по данным».
        """
        out = np.zeros(len(ts))
        for other in peers:
            if other is me:
                continue
            w = (me["width"] + other["width"]) / 2
            d_px = (y - self.to_screen(ts, np.interp(ts, other["tt"], other["vals"]))[:, 1]) / self.fr.unit
            k = np.clip(1 - np.abs(d_px) / w, 0, 1)
            k = k * k * (3 - 2 * k)                                      # плавно: smoothstep
            out += self.SPREAD * w * k * (1.0 if me["order"] < other["order"] else -1.0)
        return out


# ---------- доп. легенда: ШИМ колес ----------

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
            self.cache[k] = [g.points - ref for g in t]
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
    """«▬ левый 150 + control = 255   ▬ правый 150 − control = 45» — ШИМ физических колес (своя строка).

    Полоска — цвет оси колеса на роботе; знаки — из подключения моторов (model.wheel_signs).
    Число стоит сразу после «=»; следующая группа отстоит на ширину самого длинного числа — строка не дергается.
    opacity — ValueTracker: приглушать на время возврата между прогонами.
    """
    NUM_W = 54   # место под самое длинное число («−255»), px

    def __init__(self, fr, x_px, signs, law="control", st=Style()):
        super().__init__()
        self.fr, self.st = fr, st
        self.PX = st.text_px
        self.opacity = ValueTracker(1.0)
        y = row_baseline(fr, fr.layout.readout_y_px, st)
        self.texts, self.numbers, self.anchors = VGroup(), VGroup(), []
        x = fr.px(x_px, 0)[0]
        for name, sign, color in (("левый", signs[0], st.L), ("правый", signs[1], st.R)):
            sw = Rectangle(width=18 * fr.unit, height=9 * fr.unit, stroke_width=0, fill_color=color, fill_opacity=1)
            sw.move_to([x, y + 0.25 * st.text_px * fr.unit, 0], aligned_edge=LEFT)
            lab = text(fr, name, self.PX, st.text, st)
            lab.move_to([sw.get_right()[0] + 8 * fr.unit, 0, 0], aligned_edge=LEFT)
            frm = text(fr, f"150 {'+' if sign > 0 else '−'} {law} =", self.PX, st.muted, st)
            on_baseline(lab, y)
            frm.move_to([lab.get_right()[0] + 10 * fr.unit, 0, 0], aligned_edge=LEFT)
            on_baseline(frm, y)
            left = frm.get_right()[0] + 10 * fr.unit
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
    """Шапка + блок формулы + график с сигналами + легенда из сигналов + доп. легенда (ШИМ колес)."""

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
        if readout is not None:                           # своя строка под легендой, от левого края графика
            self.readout = MotorReadout(fr, fr.layout.margin_px, st=st, **readout)
        limit = fr.px(fr.layout.graph_w_px - fr.layout.margin_px, 0)[0]
        for row, name in ((self.legend, "легенда"), (self.readout, "строка ШИМ колес")):
            if row is not None and row.get_right()[0] > limit:
                warnings.warn(f"{name} не помещается в панель — сократите подписи")

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

    def curve(self, traj, sig, t_end, opacity=None, peers=None):
        return self.plot.curve(traj.tt, getattr(traj, sig.key), color_of(self.st, sig.color), t_end, opacity,
                               sig.width_px, peers)

    def curves(self, traj, t_end, opacity=None, signals=None):
        """Кривые сигналов панели (или signals) для траектории traj; где они совпадают — расходятся."""
        peers = []
        return VGroup(*[self.curve(traj, s, t_end, opacity, peers) for s in (signals or self.signals)])
