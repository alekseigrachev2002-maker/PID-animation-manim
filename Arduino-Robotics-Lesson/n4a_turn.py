"""Урок 4, сцена 4a — итоговая задача: от старта до финиша, на главном перекрестке X решение по дальномеру.

Робот (датчики линии + дальномер HC-SR04 в креплении) стартует с верхней внешней прямой — датчики уже за линией
старт/финиш — и едет по часовой стрелке регулятором ПИД уроков 3 и 4 (kp 0,06, kd 1 — итог 2c; ki 0,005 — из 3a).

Версии отличаются только кубиком (VERSIONS): где он стоит и какого цвета, или его нет. Все остальное выводится
из кубика: сначала робот смотрит в сторону без кубика («не угадал»), решение — ехать от кубика (нет кубика —
прямо), финиш — на перекрестке номер FINISH_AT по маршруту после решения:
    left      — зеленый кубик справа → налево → по дуге R 700 и внешней петле, стоп на 1-м перекрестке;
    right     — малиновый кубик слева → направо → через внутреннюю петлю: финиш на внутренней прямой (1-й),
                снова X (2-й), старт/финиш на внешней прямой (3-й), стоп;
    straight  — кубика нет → прямо через X → стоп на 1-м (финиш на верхней внутренней прямой).

Сюжет (Nano4a.construct) — общий список шагов: по линии до X; на перекрестке — «задержка» до центра X: моторы еще
DELAY секунд едут прямо (motors(speed, speed); delay(260);), затем motors(0, 0) — одна остановка, робот
докатывается, ось колес встает на центр X; сканирование ±SCAN_DEG по времени, провал dist ниже LIMIT_CM — кубик;
поворот на ветку на BRANCH_DEG (датчики встают на ее линию, дальше ведет регулятор) или обратно на курс; по линии
до финиша. Два вида проигрывают одни и те же шаги:
с графиком (Nano4aGraph: uP, uD, uI и сумма control, при сканировании — dist) и 16:9 без графика
(Nano4aFull: трек и робот на весь кадр, подпись шага сверху слева). Классы для Manim — циклом по VERSIONS
и видам: Nano4aLeft, Nano4aLeftFull, Nano4aRight, …

Простая езда по линии ускорена до FAST раз (атрибут сцены fast, движок) с плавным переходом; не ускоряются подъезд к X,
сканирование и решение, начало после решения, подъезд к финишу. Кратность — справа вверху.
Кубик — центр на линии ветки в CUBE_DIST от X по прямой; грань, на которую смотрит датчик, перпендикулярна линии
в центре кубика. Выбросы uD (D отвечает на скачок err, когда датчик въезжает на поперечную линию) — как есть,
за пределами шкалы кривые обрезаны краем графика.
"""
from nano_base import NanoScene, render  # первым: добавляет в путь корень репозитория

from dataclasses import dataclass

import numpy as np
from manim import UP, FadeIn, FadeOut, Line, ValueTracker, VGroup

import model
from model import bisect, end_pose, sonar_distance
from n3a_i_offset import KI, PD
from nano_track import S_FINISH_OUTER, S_X_IN, S_X_OTHER, ShortTrack
from params import Controller, Geometry, Layout, Motor
from scenes.base import RunClock, path_time
from track import Block
from visuals.layout import Frame, Z_UI, text
from visuals.panel import GraphPanel, RateLabel, Signal, TitleBar
from visuals.robot import SonarRobot

START = S_FINISH_OUTER + Geometry().sensor_y + 25.0   # верхняя внешняя прямая: датчики уже за линией старт/финиш
CUBE_DIST = 160.0     # от X до центра кубика по прямой
CUBE_SIDE = 50.0
GREEN = ("#22C55E", "#166534")
CRIMSON = ("#DB2777", "#831843")
PID = PD              # kp 0,06, kd 1 — итог 2c; ki — KI из 3a
THRESHOLD = 300.0     # порог перекрестка: оба датчика ниже — перекресток (порог 500 выброс uD на X не убирает)
DELAY = 0.26          # «задержка» до центра X, с: motors(speed, speed); delay(260); motors(0, 0) — подобрано bisect,
                      # с докатыванием ось колес встает на X (±2 мм)
LIMIT_CM = 15         # порог дальномера: кубик при сканировании — 4 см (ось на X, кубик в 16 см), без кубика — 60
SCAN_DEG = 90.0       # сканирование: на столько градусов в каждую сторону от курса остановки
BRANCH_DEG = 90.0     # поворот на ветку: ось на X, линия другой ветки — под ~90°: датчики встают на нее
MAX_CM = 60           # верх шкалы dist: «кубика нет»
U_RANGE = (-150, 300)
FAST = 3.0            # ускорение простой езды по линии (максимум); 1,0 — без ускорения
PWM_TURN = 150
SLOW_START = 0.0      # без ускорения после старта: 0 — ускорение с самого старта (плавно, по EMA)
SLOW_END = 400.0      # без ускорения: последние мм перед стопом
SLOW_AFTER = 300.0    # без ускорения: первые мм после решения
U_SIGNALS = [Signal("uP", "cP", "e"), Signal("uD", "cD", "D"), Signal("uI", "cI", "I"), Signal("control", "cU", "u", 4.0)]
DIST = Signal("dist, см", "dist", "dist")
U_FORMULA = ("control = uP + uD + uI", [("control", "u"), ("uP", "e"), ("uD", "D"), ("uI", "I")])
D_FORMULA = (f"dist < {LIMIT_CM}", [("dist", "dist")])
SIDE = {+1: "налево", -1: "направо", 0: "прямо"}
FINISH_AT = {+1: 1, -1: 3, 0: 1}   # куда поехал робот → номер финишного перекрестка по маршруту (см. докстринг)
TITLE = "Куда повернуть?"   # один для всех версий: пояснение в шапке 16:9 всегда начинается с одного места


@dataclass(frozen=True)
class CubeSpec:
    side: int             # +1 — кубик слева от робота на X (ветка вниз-влево), −1 — справа (ветка вверх-вправо)
    color: tuple          # заливка и контур


# версия → кубик (None — кубика нет); имя версии — куда поедет робот
VERSIONS = {
    "left": CubeSpec(side=-1, color=GREEN),
    "right": CubeSpec(side=+1, color=CRIMSON),
    "straight": None,
}
FILES = {"left": "scene-4a-turn-left", "right": "scene-4a-turn-right", "straight": "scene-4a-straight"}


def crossing_point(track):
    """Точка X: где маршрут пересекает сам себя (два прохода у S_X_IN и S_X_OTHER)."""
    a = np.array([track.route.pose_at(s)[:2] for s in np.arange(S_X_IN - 80, S_X_IN + 80, 1.0)])
    b = np.array([track.route.pose_at(s)[:2] for s in np.arange(S_X_OTHER - 80, S_X_OTHER + 80, 1.0)])
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return (a[i] + b[j]) / 2


def cube_pose(track, side):
    """Кубик на ветке другой линии: side +1 — левая (по часовой от S_X_OTHER), −1 — правая. Центр — на линии
    в CUBE_DIST от X по прямой; поворот — по направлению линии в центре кубика (±10 мм)."""
    x0 = crossing_point(track)
    s = S_X_OTHER
    while np.linalg.norm(np.array(track.route.pose_at(s)[:2]) - x0) < CUBE_DIST:
        s += side * 0.5
    (ax, ay, _), (bx, by, _) = track.route.pose_at(s - 10), track.route.pose_at(s + 10)
    return np.array(track.route.pose_at(s)[:2]), float(np.arctan2(-(bx - ax), by - ay))


def with_display(run):
    """Кривые для графика: control = uP + uD + uI; значения обрезаны чуть за краем шкалы (дальше — край графика)."""
    lo, hi = U_RANGE
    clip = lambda v: np.clip(v, lo - 8, hi + 8)   # noqa: E731
    run.cP, run.cD, run.cI = clip(run.uP), clip(run.uD), clip(run.uI)
    run.cU = clip(run.uP + run.uD + run.uI)
    return run


def turn_pwm(side):
    """Поворот на месте: налево (side > 0) — motors(−150, 150), направо — motors(150, −150)."""
    return (-PWM_TURN, PWM_TURN) if side > 0 else (PWM_TURN, -PWM_TURN)


class Nano4a(NanoScene):
    """Сюжет сцены для кубика self.cube (CubeSpec или None): траектории и шаги с подписями. Как показывать —
    в видах Nano4aGraph и Nano4aFull (методы ui_*)."""
    robot_cls = SonarRobot
    fast = FAST
    cube = None
    t_swap = 0.8          # смена вида между подсценами, с

    # ---------- что выводится из кубика ----------

    @property
    def cube_side(self):
        return self.cube.side if self.cube else 0

    @property
    def go_side(self):
        """Куда ехать после решения: от кубика; нет кубика — прямо (0)."""
        return -self.cube_side

    @property
    def stop_at(self):
        return FINISH_AT[self.go_side]

    def scan_order(self):
        """Сначала — в сторону без кубика («не угадал»); кубика нет — сначала налево."""
        first = -self.cube_side if self.cube_side else +1
        return first, -first

    def decision(self):
        if not self.cube:
            return "кубика нет: едем прямо"
        return f"кубик {'слева' if self.cube_side > 0 else 'справа'}: едем {SIDE[self.go_side]}"

    def look_result(self, seg2, t0, t1):
        """Итог одного взгляда: самый короткий dist за поворот — кубик (см) или «кубика нет»."""
        m = (seg2.tt >= t0) & (seg2.tt <= t1)
        d = float(seg2.dist[m].min())
        return f"кубик, {d:.0f} см" if d < LIMIT_CM else "кубика нет"

    # ---------- траектория ----------

    def drive(self, pose, pwm, t):
        return model.drive(self.track, pose, pwm, t, stop=True, geom=self.geom, sensor=self.sensor)

    def turn_to(self, pose, psi_target, side=0):
        """Поворот на месте по времени (с выбегом) до курса psi_target — в сторону side (+1 налево, −1 направо),
        side = 0 — кратчайшим поворотом."""
        d = (psi_target - pose[2] + np.pi) % (2 * np.pi) - np.pi
        if side and np.sign(d) != side:
            d += side * 2 * np.pi
        pwm = turn_pwm(np.sign(d))
        t = bisect(lambda t: abs(self.drive(pose, pwm, t).psi[-1] - pose[2]) >= abs(d), 0.02, 4.0)
        return self.drive(pose, pwm, t)

    def pid_run(self, pose, stop_at, t_on=0.0):
        """По линии до перекрестка номер stop_at; там t_on секунд прямо (motors(speed, speed); delay), затем стоп."""
        run = model.cached_simulate(self.track, tuple(map(float, pose)), ctrl=Controller(**PID), ki=KI, t_end=60.0,
                                    sensor=self.sensor, motor=Motor(), stop_below=THRESHOLD, stop_at=stop_at)
        speed = Controller().speed
        return with_display(run), model.brake(self.track, run, geom=self.geom, sensor=self.sensor, motor=Motor(),
                                              pwm=(speed, speed), t_on=t_on)

    def plan(self):
        run1, stop1 = self.pid_run(self.track.pose_at(START), 1, DELAY)
        h0 = end_pose(stop1)[2]                                    # курс остановки
        first, second = self.scan_order()
        look1 = self.turn_to(end_pose(stop1), h0 + first * np.radians(SCAN_DEG))
        look2 = self.turn_to(end_pose(look1), h0 + second * np.radians(SCAN_DEG))
        # с другой стороны — обратно через курс остановки (с ветки на ветку — на 180°, не через кубик)
        go = self.turn_to(end_pose(look2), h0 + self.go_side * np.radians(BRANCH_DEG), side=first)
        run3, stop3 = self.pid_run(end_pose(go), self.stop_at)
        return dict(run1=run1, stop1=stop1, look1=look1, look2=look2, go=go, run3=run3, stop3=stop3,
                    first=first, second=second)

    def zones(self, run, slow_start):
        """Где не ускорять: начало прогона и подъезд к стопу (время модели)."""
        s = (run.sL + run.sR) / 2
        length = float(s[-1] - s[0])
        return [(0.0, path_time(run, slow_start)), (path_time(run, length - SLOW_END), run.t[-1])]

    def dist_of(self, seg):
        """Что показывает дальномер на тиках отрезка, см (кубика нет в луче — MAX_CM)."""
        xs, ys, ps = (np.interp(seg.tt, seg.t, getattr(seg, k)) for k in ("x", "y", "psi"))
        return np.array([min(sonar_distance((x, y, a), self.track.blocks), MAX_CM * 10) / 10
                         for x, y, a in zip(xs, ys, ps)])

    # ---------- сцена ----------

    def setup(self):
        super().setup()
        self.track = ShortTrack()
        if self.cube:                                                   # кубик — предмет поля (track.blocks)
            center, angle = cube_pose(self.track, self.cube_side)
            self.track.blocks = [Block(tuple(center), CUBE_SIDE, angle, *self.cube.color)]

    def construct(self):
        p = self.plan()
        seg1 = model.chain(p["run1"], p["stop1"])
        seg2 = model.chain(p["look1"], p["look2"], p["go"])
        seg2.dist = self.dist_of(seg2)
        seg3 = model.chain(p["run3"], p["stop3"])
        self.ui_open(p, seg1, "едем к перекрестку")

        # 1: по линии до X — одна остановка
        run1 = p["run1"]
        self.begin_segment()
        self.advance_fast(run1.t[-1], self.zones(run1, SLOW_START))
        self.ui_freeze()
        self.advance(seg1.t[-1], *self.ui_caption("перекресток: доезжаем до центра, стоп"))
        self.end_segment(check=False)

        # 2: сканирование дальномером, решение, поворот
        self.play(*self.ui_phase(2, seg2, f"смотрим {SIDE[p['first']]}"), run_time=self.t_swap)
        self.switch(seg2)
        self.ui_curves(seg2)
        self.begin_segment()
        t1 = p["look1"].t[-1]
        t2 = t1 + p["look2"].t[-1]
        self.advance(t1)
        self.play(*self.ui_caption(f"смотрим {SIDE[p['first']]}: {self.look_result(seg2, 0.0, t1)}"), run_time=0.8)
        self.advance(t2, *self.ui_caption(f"смотрим {SIDE[p['second']]}"))
        self.play(*self.ui_caption(f"смотрим {SIDE[p['second']]}: {self.look_result(seg2, t1, t2)}"), run_time=0.8)
        self.play(*self.ui_caption(self.decision()), run_time=0.8)
        self.advance(seg2.t[-1])
        self.ui_freeze()
        self.end_segment(check=False)

        # 3: по линии до финиша
        run3 = p["run3"]
        self.play(*self.ui_phase(3, run3, "едем к финишу"), run_time=self.t_swap)
        self.switch(seg3)
        self.ui_curves(run3)
        self.begin_segment()
        self.advance_fast(run3.t[-1], self.zones(run3, SLOW_AFTER))
        self.ui_freeze()
        self.advance(seg3.t[-1], *self.ui_caption(f"финиш: стоп на {self.stop_at}-м перекрестке"))
        self.end_segment(check=False)
        self.wait(0.5, frozen_frame=False)

    # ---------- показ: переопределяют виды ----------

    def ui_open(self, p, seg1, caption):
        raise NotImplementedError

    def ui_caption(self, caption):
        raise NotImplementedError

    def ui_phase(self, n, traj, caption):
        return self.ui_caption(caption)

    def ui_curves(self, traj):
        pass

    def ui_freeze(self):
        pass


class Nano4aGraph(Nano4a):
    """Вид с панелью графиков слева (как остальные сцены курса): u-кривые, при сканировании — dist."""
    t_swap = 1.2

    def u_panel(self, run, caption):
        T = run.t[-1]
        return GraphPanel(self.fr, self.st, TITLE, caption, y=("u", *U_RANGE), t_max=T * 1.04,
                          t_ticks=np.arange(0, T, 2.0 if T > 12 else 1.0), signals=U_SIGNALS,
                          formula=U_FORMULA[0], formula_colors=U_FORMULA[1])

    def crossing_marks(self, run, clock):
        """Метки перекрестков (фронт «оба ниже порога») на оси времени с номерами; появляются, когда до них дошли."""
        cross = (run.L < THRESHOLD) & (run.R < THRESHOLD)
        times = list(run.tt[1:][cross[1:] & ~cross[:-1]])
        if len(times) < self.stop_at:                 # перекресток финиша: на нем прогон кончился (стоп)
            times.append(run.t[-1])
        marks = VGroup()
        plot, fr, st = self.panel.plot, self.fr, self.st
        for n, t in enumerate(times, 1):
            a, b = plot.to_screen([t, t], list(U_RANGE))
            tick = Line(a, b, color=st.muted, stroke_width=fr.stroke(1))
            lab = text(fr, str(n), st.small_px, st.text, st).next_to(b, UP, buff=4 * fr.unit)
            m = VGroup(tick, lab).set_z_index(Z_UI)
            m.add_updater(lambda m, t=t: m.set_opacity(1.0 if clock() >= t else 0.0))
            marks.add(m)
        return marks

    def ui_open(self, p, seg1, caption):
        self.show(self.track, seg1)
        self.rate = RateLabel(self.fr, self.st)
        run1 = p["run1"]
        self.panel = self.u_panel(run1, caption)
        self.header = self.panel.header
        self.clock = RunClock(self.t)
        curves = self.panel.curves(run1, self.clock)
        self.drawn = VGroup(self.panel.axes, self.panel.legend, curves)   # что погаснет при смене графика
        self.add(self.panel, curves, self.rate)
        self.play(self.panel.intro(), run_time=0.8)

    def ui_caption(self, caption):
        return [self.header.set_caption(caption)]

    def ui_phase(self, n, traj, caption):
        """Смена графика между подсценами: старые оси, легенда, кривые гаснут, новые появляются; шапка — общая."""
        old = self.drawn
        if n == 2:
            self.panel = GraphPanel(self.fr, self.st, "·", "·", y=("dist, см", 0, MAX_CM), t_max=traj.t[-1] * 1.04,
                                    t_ticks=np.arange(0, traj.t[-1], 0.5), signals=[DIST])   # шапка не нужна — общая
            limit = self.panel.plot.hline(LIMIT_CM, f"{LIMIT_CM} см", ValueTracker(1.0), side="left")
            self.drawn = VGroup(self.panel.axes, self.panel.legend, limit)
            formula = D_FORMULA
        else:
            self.panel = self.u_panel(traj, "·")
            self.drawn = VGroup(self.panel.axes, self.panel.legend)
            formula = U_FORMULA
        return [FadeOut(old), FadeIn(self.drawn),
                self.header.set_text(formula[0], self.panel._colors(formula[1]), caption)]

    def ui_curves(self, traj):
        self.clock = RunClock(self.t)
        curves = self.panel.curves(traj, self.clock)
        if hasattr(traj, "cU"):                                         # по линии — с метками перекрестков
            curves.add(self.crossing_marks(traj, self.clock))
        self.drawn.add(curves)
        self.add(curves)

    def ui_freeze(self):
        self.clock.stop()


class Nano4aFull(Nano4a):
    """Вид 16:9 без графика: трек и робот на весь кадр, сверху — шапка-полоса (TitleBar) с заголовком, пояснением
    шага и кратностью ускорения справа."""
    panel_bg = False                                                       # без темной панели графиков

    def ui_open(self, p, seg1, caption):
        self.show(self.track, seg1, anchor_px=(640.0, 420.0))           # робот ниже середины: впереди место
        self.bar = TitleBar(self.fr, TITLE, caption, self.st)
        self.rate = RateLabel(self.fr, self.st, right_px=self.fr.layout.width_px - self.fr.layout.margin_px,
                              color=self.st.text, centered=True)
        self.add(self.bar, self.rate)
        self.play(self.bar.intro(), run_time=0.8)

    def ui_caption(self, caption):
        return [self.bar.set_caption(caption)]


# классы для Manim: версия × вид — Nano4aLeft, Nano4aLeftFull, Nano4aRight, …
SCENES = {}
for _name, _cube in VERSIONS.items():
    for _full, _base in ((False, Nano4aGraph), (True, Nano4aFull)):
        _cls = type(f"Nano4a{_name.capitalize()}{'Full' if _full else ''}", (_base,),
                    {"cube": _cube, "__module__": __name__, "__doc__": f"Сцена 4a, версия «{_name}»."})
        SCENES[_name, _full] = globals()[_cls.__name__] = _cls


class Nano4aCloseUp(SCENES["left", False]):
    """Картинка: крупный план крепления дальномера на шасси и кубик (масштаб ×2,5 к сценам). Только последний кадр."""

    def setup(self):
        super().setup()
        self.fr = Frame(Layout(px_per_mm=4.0))

    def construct(self):
        x, y = crossing_point(self.track)
        psi = self.track.route.pose_at(S_X_OTHER)[2] + np.pi
        traj = model.prescribed(self.track, np.array([0.0, 0.01]), x, y, psi, sensor=self.sensor)
        center = np.array([x, y]) + np.array([-np.sin(psi), np.cos(psi)]) * 70
        self.show(self.track, traj, camera="fixed", center=center, anchor_px=(1066.5, 360.0), rot=-psi)
        self.wait(0.1, frozen_frame=False)


if __name__ == "__main__":
    for name in VERSIONS:
        render(SCENES[name, True], "480p")
    # render(SCENES["left", True], "480p")   # одна версия: "left", "right" или "straight"
    # render(SCENES["left", False], "480p")  # вид с графиком
    # render(Nano4aCloseUp, "720p", still=True)
