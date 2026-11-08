"""Трасса: осевые линии, расстояние до них и конструктор. Без Manim.

Трасса строится «черепашкой»: старт, курс, затем прямые и повороты.

    track = Track(
        Route(start=(0, -1500), heading="вверх").forward(1500).mark("поворот").right(R=100).forward(1500),
        Route(start=(-600, 400), heading="вправо").forward(1200),        # поперечная линия перекрёстка
    )
    start = track.pose("поворот", -350)                                  # поза робота за 350 мм до поворота

Внутри осевая — ломаная с шагом 0.5 мм; радиус поворота существует только при построении.
"""
import numpy as np
from scipy.spatial import cKDTree

STEP = 0.5  # шаг дискретизации осевой, мм

# курс ψ отсчитывается от +y против часовой стрелки (как у робота)
HEADINGS = {"вверх": 0.0, "влево": 90.0, "вниз": 180.0, "вправо": -90.0}


class Route:
    """Одна линия трассы, собранная командами forward / right / left."""

    def __init__(self, start=(0.0, 0.0), heading="вверх"):
        self.pts = [np.asarray(start, float)]
        self.psi = np.radians(HEADINGS.get(heading, heading))
        self.marks = {}

    def _dir(self):
        return np.array([-np.sin(self.psi), np.cos(self.psi)])

    def forward(self, length):
        n = max(int(np.ceil(length / STEP)), 1)
        p0 = self.pts[-1]
        self.pts += [p0 + self._dir() * s for s in np.linspace(0, length, n + 1)[1:]]
        return self

    def _turn(self, R, angle_deg, side):
        """side = +1 — направо (по часовой), −1 — налево."""
        p0, a = self.pts[-1], np.radians(angle_deg)
        right = np.array([np.cos(self.psi), np.sin(self.psi)])     # нормаль вправо от курса
        c = p0 + side * R * right
        n = max(int(np.ceil(a * R / STEP)), 1)
        for th in np.linspace(0, a, n + 1)[1:]:
            rot = -side * th                                      # направо — вращение по часовой
            v = p0 - c
            self.pts.append(c + np.array([v[0] * np.cos(rot) - v[1] * np.sin(rot),
                                          v[0] * np.sin(rot) + v[1] * np.cos(rot)]))
        self.psi -= side * a
        return self

    def right(self, R, angle=90):
        return self._turn(R, angle, +1)

    def left(self, R, angle=90):
        return self._turn(R, angle, -1)

    def mark(self, name):
        """Запомнить текущую точку линии под именем (например, начало поворота)."""
        self.marks[name] = len(self.pts) - 1
        return self


class Line:
    """Осевая линия: ломаная (N, 2) в мм."""

    def __init__(self, points, marks=None):
        self.points = np.asarray(points, float)
        self.s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))])
        self.marks = {k: self.s[i] for k, i in (marks or {}).items()}   # имя → длина вдоль линии, мм
        self._tree = cKDTree(self.points)

    def _nearest(self, flat):
        """Ближайший отрезок осевой для каждой точки: (начало, конец, расстояние)."""
        _, i = self._tree.query(flat)
        n = len(self.points)
        prev, nxt = np.maximum(i - 1, 0), np.minimum(i + 1, n - 1)
        d1 = _seg_dist(flat, self.points[prev], self.points[i])
        d2 = _seg_dist(flat, self.points[i], self.points[nxt])
        first = d1 <= d2
        a = np.where(first[:, None], self.points[prev], self.points[i])
        b = np.where(first[:, None], self.points[i], self.points[nxt])
        return a, b, np.minimum(d1, d2)

    def distance(self, p) -> np.ndarray:
        """Расстояние от точек p (..., 2) до осевой."""
        p = np.asarray(p, dtype=float)
        return self._nearest(p.reshape(-1, 2))[2].reshape(p.shape[:-1])

    def side(self, p) -> np.ndarray:
        """С какой стороны осевой лежат точки: +1 слева по ходу линии, −1 справа."""
        p = np.asarray(p, dtype=float)
        flat = p.reshape(-1, 2)
        a, b, _ = self._nearest(flat)
        ab, ap = b - a, flat - a
        return np.sign(ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]).reshape(p.shape[:-1])

    def pose_at(self, s):
        """Поза (x, y, ψ) на линии на расстоянии s от её начала: точка и курс вдоль линии."""
        i = int(np.clip(np.searchsorted(self.s, s), 1, len(self.s) - 1))
        a, b = self.points[i - 1], self.points[i]
        f = (s - self.s[i - 1]) / max(self.s[i] - self.s[i - 1], 1e-9)
        x, y = a + (b - a) * f
        return float(x), float(y), float(np.arctan2(-(b - a)[0], (b - a)[1]))


def _seg_dist(p, a, b):
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.where(L2 > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1), 0)
    t = np.clip(t, 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


class Track:
    """Трасса — набор линий. Принимает Route (собирается здесь) или готовые Line.

    squares — разметка поля без влияния на датчики: квадраты (x0, y0, x1, y1), мм.
    """

    def __init__(self, *items, squares=()):
        self.lines = [Line(np.array(it.pts), it.marks) if isinstance(it, Route) else it for it in items]
        self.squares = list(squares)

    def distances(self, p) -> list:
        return [ln.distance(p) for ln in self.lines]

    def pose(self, mark, offset=0.0):
        """Поза на линии со смещением offset (мм) от метки mark; offset < 0 — раньше метки."""
        for ln in self.lines:
            if mark in ln.marks:
                return ln.pose_at(ln.marks[mark] + offset)
        raise KeyError(mark)


def field1() -> Track:
    """Поле 2400×1500 с прямыми линиями (PDF): начало координат — левый нижний угол поля, y вверх.

    Чёрные линии 30 мм на сетке 300 мм, перекрёстки под 90°; жёлтые квадраты 300×300 с перекрёстком в центре.
    """
    X = (601.7, 901.6, 1201.6, 1501.5, 1801.5)          # вертикали
    v = [Route(start=(x, 150.1), heading="вверх").forward(1199.8) for x in X]
    h = [Route(start=(450.0, 300.1), heading="вправо").forward(1500.0),
         Route(start=(450.0, 1199.9), heading="вправо").forward(1500.0),
         Route(start=(50.0, 750.0), heading="вправо").forward(2299.7)]           # центральная, от старта до контроля
    crosses = [Route(start=(200.0, 600.0), heading="вверх").forward(300.0),      # «СТАРТ»
               Route(start=(2199.7, 600.0), heading="вверх").forward(300.0)]     # «КОНТРОЛЬ»
    squares = [(x - 150.0, y0, x + 150.0, y0 + 300.0) for x in X for y0 in (150.1, 1050.0)]
    squares += [(50.0, 600.0, 350.0, 900.0), (2049.7, 600.0, 2349.7, 900.0)]
    return Track(*v, *h, *crosses, squares=squares)
