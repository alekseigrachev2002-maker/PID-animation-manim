"""Основа сцен нового курса: стенд олимпиадных сцен (scenes/base.py) с датчиками и треком Short Track.

Отличия от олимпиадных сцен:
- датчики — nano_track.SENSOR (белое 900, черное 200, линия 20 мм);
- камера повернута так, что на старте сцены робот едет снизу вверх (поворот на всю сцену, не следит за курсом);
- на поле — черные круги трека (Decor);
- подсцена — MOTION = 8 с движения (прогон с докатыванием, маневр); заголовок, смена подписи и перестановка
  «рукой» идут сверх этих 8 с и в счет не входят; конец каждой подсцены записывается в SEGMENTS (для картинок);
- прогоны регулятора — nano_model (с И); длина прогона подбирается под 8 с (fit_run); потерявший линию робот
  едет дальше;
- «рука» везде с одной скоростью: время перестановки — от расстояния и угла (HAND_V, HAND_W);
- простая езда по линии может быть ускорена (FAST, advance_fast) с плавным переходом и пометкой кратности.
Замедление (PLAYBACK — езда, HAND — «рука») и панель графиков — общие с олимпиадными сценами.
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from manim import LEFT, Circle, FadeOut, VGroup, VMobject, config  # noqa: E402

import model  # noqa: E402
import nano_model  # noqa: E402
from nano_track import SENSOR  # noqa: E402
from params import Controller, Motor  # noqa: E402
from scenes.base import HAND, PLAYBACK, PIDScene, RunClock, RunSeries, render as _render  # noqa: E402,F401
from visuals.camera import Camera  # noqa: E402
from visuals.field import Field  # noqa: E402
from visuals.layout import Z_UI, baseline, text as label_text  # noqa: E402
from visuals.panel import GraphPanel  # noqa: E402
from visuals.robot import Robot  # noqa: E402

I_COLOR = "#F87171"          # И: красный — отличается от розового D, при дальтонизме теплее зеленого P
Z_DECOR = 0.2                # над линиями поля (0), под колесами (1)
MOTION = 8.0                 # движение в подсцене, с видео (прогон с докатыванием, маневр)
SEGMENTS = {}                # имя сцены → моменты конца подсцен в видео, с (заполняется при рендере)
FAST = 3.0                   # ускорение простой езды по линии (сцена 4a), максимум; 1,0 — без ускорения
FAST_TAU = 0.5               # плавность перехода ускорения: постоянная EMA, с времени модели
HAND_V = 150.0               # «рука»: средняя скорость переноса, мм/с (время модели; на экране — с замедлением HAND)
HAND_W = 120.0               # «рука»: средняя скорость поворота, °/с
HAND_MIN = 0.4               # самая короткая перестановка, с


def hand_time(a, b):
    """Время перестановки «рукой» из позы a в позу b (сдвиг и поворот вместе), с — одна скорость для всех сцен."""
    dpsi = abs((b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi)
    return max(np.hypot(b[0] - a[0], b[1] - a[1]) / HAND_V, np.degrees(dpsi) / HAND_W, HAND_MIN)


def bisect(done, lo, hi, n=30):
    """Наименьшее x в [lo, hi], при котором done(x) истинно (done монотонна): подбор времени команды, угла."""
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if done(mid) else (mid, hi)
    return (lo + hi) / 2


def fit_run(simulate, brake, target):
    """Прогон run = simulate(t_end), который вместе с докатыванием brake(run) длится target секунд модели
    (подстройка t_end по перебору/недобору, до 6 шагов). Возвращает (run, stop)."""
    t_end = target - 0.25
    for _ in range(6):
        run = simulate(round(t_end, 3))
        stop = brake(run)
        over = run.t[-1] + stop.t[-1] - target
        if abs(over) <= 0.01:
            break
        t_end -= over + 0.005
    return run, stop


def rotation(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def warp(t0, t1, zones, fast=None, tau=FAST_TAU, speed=PLAYBACK, dt=0.002):
    """Ускорение проигрывания на участке модели [t0, t1]: в зонах zones = [(a, b), ...] (время модели) кратность 1,
    вне их — fast. Переход плавный: EMA вперед и назад по времени модели, берется меньшая — так к началу зоны
    кратность уже 1 и после зоны растет постепенно. Возвращает (длительность видео, rate_func для t, τ, k(τ))."""
    fast = FAST if fast is None else fast
    tau_grid = np.arange(t0, t1 + dt / 2, dt)
    target = np.full(len(tau_grid), float(fast))
    for a, b in zones:
        target[(tau_grid >= a) & (tau_grid <= b)] = 1.0
    alpha = 1 - np.exp(-dt / tau)
    fwd, bwd = target.copy(), target.copy()
    for i in range(1, len(target)):
        fwd[i] = fwd[i - 1] + (target[i] - fwd[i - 1]) * alpha
    for i in range(len(target) - 2, -1, -1):
        bwd[i] = bwd[i + 1] + (target[i] - bwd[i + 1]) * alpha
    k = np.maximum(np.minimum(fwd, bwd), 1.0)
    video = np.concatenate([[0.0], np.cumsum(dt / (speed * (k[1:] + k[:-1]) / 2))])
    total = float(video[-1])

    def rate(alpha_):
        return float((np.interp(alpha_ * total, video, tau_grid) - t0) / max(t1 - t0, 1e-9))
    return total, rate, tau_grid, k


def path_time(traj, dist):
    """Время модели, когда середина оси прошла dist мм от начала траектории (по пути колес)."""
    s = (traj.sL + traj.sR) / 2 - (traj.sL[0] + traj.sR[0]) / 2
    return float(np.interp(dist, s, traj.t))


class RateLabel(VGroup):
    """Пометка ускорения «×4,0» справа в строке заголовка: текущая кратность, один знак после запятой;
    при ×1,0 гаснет. Слоты под символы — как visuals.panel.Number (состав объектов не меняется)."""
    SLOTS = 5

    def __init__(self, fr, st, right_px=None):
        super().__init__(*[VMobject(fill_color=st.muted, fill_opacity=1, stroke_width=0) for _ in range(self.SLOTS)])
        self.fr, self.st, self.cache = fr, st, {}
        right_px = fr.layout.graph_w_px - fr.layout.margin_px if right_px is None else right_px   # край панели
        self.right = fr.px(right_px, 0)[0]
        ref = label_text(fr, "Кубик", st.title_px, st.text, st, weight=st.title_weight)
        ref.move_to(fr.px(fr.layout.margin_px, fr.layout.header_y_px), aligned_edge=LEFT)
        self.base = baseline(ref)
        self.set_z_index(Z_UI)
        self.set_value(1.0)

    def _glyphs(self, s):
        if s not in self.cache:
            t = label_text(self.fr, s, self.st.text_px, self.st.muted, self.st)
            ref = np.array([t.get_right()[0], baseline(t), 0.0])
            self.cache[s] = [g.points - ref for g in t]
        return self.cache[s]

    def set_value(self, k):
        s = f"×{k:.1f}".replace(".", ",")
        glyphs = self._glyphs(s)
        anchor = np.array([self.right, self.base, 0.0])
        for j, slot in enumerate(self):
            if j < len(glyphs):
                slot.set_points(glyphs[j] + anchor)
            else:
                slot.clear_points()
        self.set_fill(opacity=float(np.clip((k - 1.0) / 0.15, 0, 1)))
        return self


class RotCamera(Camera):
    """Камера, повернутая на угол rot (рад, против часовой): поле на экране повернуто, робот — тоже."""

    def __init__(self, fr, mode="follow", center=(0.0, 0.0), anchor=None, rot=0.0):
        super().__init__(fr, mode, center, anchor)
        self.rot = float(rot)
        self.R = rotation(self.rot)

    def to_screen(self, pts_mm):
        p = (np.asarray(pts_mm, float).reshape(-1, 2) - self.center) @ self.R.T * self.fr.mm
        return np.column_stack([p, np.zeros(len(p))]) + self.anchor


class RotField(Field):
    """Поле для повернутой камеры: сдвиг вслед за камерой — тоже повернутый."""

    def update_view(self):
        d = self.camera.R @ (self.drawn - self.camera.center) * self.camera.fr.mm
        if np.any(d):
            self.shift([d[0], d[1], 0])
            self.drawn = self.camera.center.copy()
        return self


class Decor(VGroup):
    """Черные круги трека (на датчики не влияют: маршрут их не задевает). Двигаются вслед за камерой."""

    def __init__(self, view, discs, st):
        super().__init__()
        self.view = view
        self.centers = [np.asarray(c, float) for c, _ in discs]
        for c, r in discs:
            self.add(Circle(radius=r * view.fr.mm, stroke_width=0, fill_color=st.line, fill_opacity=1))
        self.set_z_index(Z_DECOR)

    def update_state(self, state):
        for m, c in zip(self, self.centers):
            m.move_to(self.view.point(*c))
        return self


class NanoScene(PIDScene):
    """PIDScene с датчиками нового курса, кругами трека и камерой, повернутой под стартовый курс робота.

    robot_cls — класс робота (урок 4: с дальномером); field_items(view) — свои элементы поля (кубик).
    """
    robot_cls = Robot

    def field_items(self, view):
        return []

    def setup(self):
        super().setup()
        self.sensor = SENSOR

    def begin_segment(self):
        """Начало движения подсцены (после заголовка, смены подписи, перестановки «рукой»)."""
        self._seg_start = getattr(self, "frames", 0) / config.frame_rate

    def end_segment(self, check=True):
        """Конец подсцены: момент — в SEGMENTS (картинки урока берут кадр отсюда); если движение не 8 с —
        предупреждение (check=False — длина задана самим движением: качание «рукой», оборот на угол)."""
        now = getattr(self, "frames", 0) / config.frame_rate
        SEGMENTS.setdefault(type(self).__name__, []).append(now)
        took = now - getattr(self, "_seg_start", now)
        if check and abs(took - MOTION) > 0.3:
            warnings.warn(f"{type(self).__name__}: движение в подсцене {len(SEGMENTS[type(self).__name__])} — "
                          f"{took:.1f} с (MOTION = {MOTION:g} с)")

    def advance_fast(self, t_to, zones, *extra, fast=None):
        """Проиграть траекторию до t_to с ускорением вне зон zones (см. warp); пометка self.rate (RateLabel),
        если есть, показывает текущую кратность."""
        t0 = self.t.get_value()
        total, rate, tau, k = warp(t0, t_to, zones, fast, speed=self.speed)
        label = getattr(self, "rate", None)
        if label is not None:
            label.add_updater(lambda m: m.set_value(float(np.interp(self.t.get_value(), tau, k))))
        self.play(self.t.animate.set_value(t_to), *extra, run_time=total, rate_func=rate)
        if label is not None:
            label.clear_updaters()
            label.set_value(1.0)

    def hand_to(self, target, *extra):
        """Переставить робота «рукой» в позу target (сдвиг и поворот вместе, колеса не крутятся) — с одной
        скоростью для всех сцен; extra — анимации на это же время (смена подписи)."""
        a = self.end_pose()
        dpsi = (target[2] - a[2] + np.pi) % (2 * np.pi) - np.pi
        t = hand_time(a, target)
        self.switch(model.move(self.track, a, (target[0], target[1], a[2] + dpsi), t, roll=False,
                               geom=self.geom, sensor=self.sensor))
        self.advance(t, *extra, speed=HAND)

    def show(self, track, traj, camera="follow", center=(0.0, 0.0), anchor_px=None, sensors=True, items=(),
             rot=None):
        """Как PIDScene.show; rot — поворот камеры (по умолчанию: стартовый курс робота смотрит вверх)."""
        self.track, self.traj = track, traj
        rot = -float(traj.psi[0]) if rot is None else rot
        self.view = RotCamera(self.fr, camera, center, None if anchor_px is None else self.fr.px(*anchor_px), rot)
        self.field = RotField(self.view, track, self.sensor, self.st)
        self.robot = self.robot_cls(self.fr, self.geom, self.st, self.sensor, sensors)
        decor = [Decor(self.view, track.discs, self.st)] if hasattr(track, "discs") else []
        decor += self.field_items(self.view)
        self.items = decor + [m for m in items if m is not None]
        self.s_off = np.zeros(2)

        def follow(_):
            s = self.state()
            self.view.follow(s["x"], s["y"])
            self.field.update_view()
            self.robot.place(self.view.point(s["x"], s["y"]), s["psi"] + self.view.rot)
            self.robot.set_wheel_phase(s["sL"], s["sR"]).set_signals(*s["levels"])
            for m in self.items:
                m.update_state(s)

        self.robot.add_updater(follow)
        follow(self.robot)
        for m in (self.field, *self.items):
            m.add_updater(lambda m: None)
        self.add(self.field, *decor, self.robot, *[m for m in self.items if m not in decor])


class NanoRunSeries(NanoScene, RunSeries):
    """Серия прогонов регулятора (как RunSeries) на треке Short Track; регулятор — с И.

    runs = [(dict(kp=0.1, kd=1.0, ki=0.002), "подпись"), ...]; одно число — только P (kd = 0).
    Прогон + докатывание после motors(0, 0) — MOTION = 8 с видео. Потерявший линию робот едет дальше (как
    настоящий). Между прогонами робота возвращают «рукой» (hand_to), кривые гаснут и подпись меняется в это же
    время — сверх 8 с.
    """
    t_on = 0.4            # строка ШИМ снова яркая

    def controller(self, ctrl):
        kw = {"kp": ctrl, "kd": 0.0} if np.isscalar(ctrl) else dict(ctrl)
        extra = {k: kw.pop(k) for k in ("ki",) if k in kw}
        return Controller(**kw), extra

    def simulate(self, pose, ctrl, t_end):
        c, extra = self.controller(ctrl)
        return nano_model.cached_simulate(self.track, tuple(float(v) for v in pose), ctrl=c, t_end=round(t_end, 3),
                                          lost_dist=np.inf, **extra)

    def brake(self, run):
        return model.brake(self.track, run, geom=self.geom, sensor=self.sensor, motor=Motor())

    def fit_run(self, pose, ctrl, video=MOTION):
        """Прогон, который вместе с докатыванием занимает video секунд видео."""
        return fit_run(lambda t: self.simulate(pose, ctrl, t), self.brake, video * self.speed)[0]

    def plan(self):
        """Прогоны всех подсцен заранее: шкала времени графика — по самому длинному."""
        return [self.fit_run(self.start(i), run[0]) for i, run in enumerate(self.runs)]

    def panel(self, t_max):
        ticks = [t for t in np.arange(0, t_max, 0.5)]
        return GraphPanel(self.fr, self.st, self.phrase, self.runs[0][1], self.y, t_max, ticks, self.signals,
                          readout=dict(signs=model.wheel_signs(Motor()), law=self.law),
                          formula=self.formula, formula_colors=self.formula_colors)

    def construct(self):
        self.track = self.make_track()
        runs = self.plan()
        panel = self.panel(max(r.t[-1] for r in runs) * 1.06)
        self.show(self.track, runs[0], items=[panel.readout])
        self.add(panel)
        self.play(panel.intro(), run_time=0.8)
        readout, curves = panel.readout, None
        for i, ((ctrl, caption, *text), run) in enumerate(zip(self.runs, runs)):   # прогон — подсцена
            if i > 0:
                self.hand_to(self.start(i), FadeOut(curves), panel.set_text(*text, caption=caption))
                self.switch(run)
                self.play(readout.opacity.animate.set_value(1.0), run_time=self.t_on)
            self.begin_segment()
            clock = RunClock(self.t)
            curves = panel.curves(self.traj, clock)
            self.add(curves)
            self.advance(self.traj.t[-1])
            clock.stop()
            self.switch(self.brake(self.traj))
            self.advance(self.traj.t[-1], readout.opacity.animate.set_value(0.35))
            self.end_segment()


def render(scene_cls, quality="480p", still=False, preview=True):
    """Как scenes.base.render: видео в videos/<Сцена>_<качество>.mp4 (общая папка с олимпиадными сценами)."""
    _render(scene_cls, quality, still, preview)
