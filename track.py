"""Трассы: осевые линии и расстояние до них. Без Manim.

Трасса — набор осевых линий (ломаных). Осевые нужны и визуалу,
поэтому хранятся как массивы точек (N, 2) в мм.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

STEP = 0.5  # шаг дискретизации осевой, мм


@dataclass
class Line:
    points: np.ndarray
    closed: bool = False
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self._tree = cKDTree(self.points)

    def _nearest(self, flat):
        """Ближайший отрезок осевой для каждой точки: (начало, конец, расстояние)."""
        _, i = self._tree.query(flat)
        n = len(self.points)
        prev = (i - 1) % n if self.closed else np.maximum(i - 1, 0)
        nxt = (i + 1) % n if self.closed else np.minimum(i + 1, n - 1)
        d1 = _seg_dist(flat, self.points[prev], self.points[i])
        d2 = _seg_dist(flat, self.points[i], self.points[nxt])
        first = d1 <= d2
        a = np.where(first[:, None], self.points[prev], self.points[i])
        b = np.where(first[:, None], self.points[i], self.points[nxt])
        return a, b, np.minimum(d1, d2)

    def distance(self, p) -> np.ndarray:
        """Расстояние от точек p (..., 2) до осевой: ближайшая вершина + проекция на соседние отрезки."""
        p = np.asarray(p, dtype=float)
        return self._nearest(p.reshape(-1, 2))[2].reshape(p.shape[:-1])

    def side(self, p) -> np.ndarray:
        """С какой стороны осевой лежат точки: +1 слева по ходу линии, −1 справа."""
        p = np.asarray(p, dtype=float)
        flat = p.reshape(-1, 2)
        a, b, _ = self._nearest(flat)
        ab, ap = b - a, flat - a
        return np.sign(ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]).reshape(p.shape[:-1])


def _seg_dist(p, a, b):
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.where(L2 > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1), 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


@dataclass
class Track:
    lines: list

    def distances(self, p) -> list:
        return [ln.distance(p) for ln in self.lines]


# ---------- формы ----------

def _segment(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.ceil(np.linalg.norm(b - a) / STEP)), 1)
    return a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a)


def _arc(c, r, a0, a1):
    n = max(int(np.ceil(abs(a1 - a0) * r / STEP)), 1)
    a = np.linspace(a0, a1, n, endpoint=False)
    return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])


def straight(y0=-300.0, y1=3000.0, x=0.0) -> Track:
    """Вертикальная прямая вдоль +y."""
    pts = np.vstack([_segment((x, y0), (x, y1)), [[x, y1]]])
    return Track([Line(pts)])


def rounded_rect(w, h, radii) -> Track:
    """Прямоугольник w×h с центром в нуле, скругления radii = (ПВ, ЛВ, ЛН, ПН).

    Обход против часовой стрелки (повороты влево), старт — середина правой стороны.
    """
    rTR, rTL, rBL, rBR = radii
    X, Y = w / 2, h / 2
    pi = np.pi
    parts = [
        _segment((X, 0), (X, Y - rTR)),
        _arc((X - rTR, Y - rTR), rTR, 0, pi / 2),
        _segment((X - rTR, Y), (-X + rTL, Y)),
        _arc((-X + rTL, Y - rTL), rTL, pi / 2, pi),
        _segment((-X, Y - rTL), (-X, -Y + rBL)),
        _arc((-X + rBL, -Y + rBL), rBL, pi, 3 * pi / 2),
        _segment((-X + rBL, -Y), (X - rBR, -Y)),
        _arc((X - rBR, -Y + rBR), rBR, 3 * pi / 2, 2 * pi),
        _segment((X, -Y + rBR), (X, 0)),
    ]
    return Track([Line(np.vstack(parts), closed=True)])


def crossroad(y_cross=400.0, y0=-300.0, y1=3000.0, half=600.0) -> Track:
    """Прямая вдоль +y и поперечная линия на y = y_cross."""
    main = straight(y0, y1).lines[0]
    cross = np.vstack([_segment((-half, y_cross), (half, y_cross)), [[half, y_cross]]])
    return Track([main, Line(cross)])
