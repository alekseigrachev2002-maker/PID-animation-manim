"""Макет кадра: перевод пикселей эталонного кадра и мм в единицы Manim, текст, фон.

Перевод единиц — только здесь (Frame). Пиксели — эталонного кадра 1280×720, начало в левом верхнем углу.
"""
import numpy as np
from manim import UP, DOWN, Rectangle, Text, VGroup, config

from params import Layout, Style

Z_PANEL = 10   # панель графиков закрывает поле
Z_UI = 12      # все, что на панели


class Frame:
    def __init__(self, layout=Layout()):
        self.layout = layout
        self.unit = config.frame_width / layout.width_px            # единиц Manim на пиксель
        self.height_px = layout.width_px * config.frame_height / config.frame_width
        self.mm = layout.px_per_mm * self.unit                       # единиц Manim на мм

    def px(self, x, y):
        return np.array([(x - self.layout.width_px / 2) * self.unit, (self.height_px / 2 - y) * self.unit, 0.0])

    def to_px_x(self, x_units):
        return x_units / self.unit + self.layout.width_px / 2

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
        """Точка экрана, где камера держит середину оси робота."""
        return self.px(*self.layout.robot_px)


_HAVE = None


def pick_font(candidates):
    """Первый шрифт из списка, который есть в системе (Segoe UI и Consolas — на Windows; на Linux — замена)."""
    global _HAVE
    if isinstance(candidates, str):
        return candidates
    if _HAVE is None:
        import manimpango
        _HAVE = set(manimpango.list_fonts())
    return next((f for f in candidates if f in _HAVE), candidates[-1])


def text(fr, s, px, color, st=Style(), font=None, **kw):
    """Текст шрифтом стиля (или font); знак умножения «·» опущен на середину строчных букв."""
    font = pick_font(font or st.font)
    t = Text(s, font=font, font_size=fr.font(px), color=color, **kw)
    glyphs = s.replace(" ", "")
    if "·" in glyphs and len(glyphs) == len(t):
        dy = _dot_lift(font, fr.font(px), kw.get("weight", "NORMAL"))
        for i, ch in enumerate(glyphs):
            if ch == "·":
                t[i].shift(dy * DOWN)
    return t


_DOT_LIFT = {}


def _dot_lift(font, size, weight):
    """На сколько центр «·» выше середины строчной «e» при этом шрифте и кегле."""
    key = (font, size, weight)
    if key not in _DOT_LIFT:
        ref = Text("e·e", font=font, font_size=size, weight=weight)
        e, dot = ref[0], ref[1]
        _DOT_LIFT[key] = dot.get_center()[1] - (e.get_bottom()[1] + e.height / 2)
    return _DOT_LIFT[key]


def baseline(t):
    """Линия шрифта текста: медиана нижних краев символов (хвосты «р», «у» и знаки «=», «−» не сбивают)."""
    return float(np.median([g.get_bottom()[1] for g in t]))


def on_baseline(t, y):
    """Поставить текст так, чтобы его линия шрифта прошла по высоте y."""
    return t.shift((y - baseline(t)) * UP)


def backdrop(fr, st=Style(), panel=True):
    """Фон кадра: поле на весь кадр и (panel=True) темная панель графиков слева."""
    field = Rectangle(width=config.frame_width, height=config.frame_height, fill_color=st.field,
                      fill_opacity=1, stroke_width=0).set_z_index(-10)
    if not panel:
        return VGroup(field)
    w = fr.layout.graph_w_px * fr.unit
    panel = Rectangle(width=w, height=config.frame_height, fill_color=st.graph_bg, fill_opacity=1,
                      stroke_width=0).move_to(fr.px(fr.layout.graph_w_px / 2, fr.height_px / 2))
    return VGroup(field, panel.set_z_index(Z_PANEL))
