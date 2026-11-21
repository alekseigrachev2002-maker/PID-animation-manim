"""Основа сцен нового курса: стенд олимпиадных сцен (scenes/base.py) с датчиками и треком Short Track.

Отличия от олимпиадных сцен:
- датчики — nano_track.SENSOR (белое 900, черное 200, линия 20 мм);
- камера повернута так, что на старте сцены робот едет снизу вверх (поворот на всю сцену, не следит за курсом);
- на поле — черные круги трека (Decor);
- промежуток сцены — BLOCK = 10 с (у олимпиадных сцен 11 с);
- прогоны регулятора — nano_model (с И); длина прогона подбирается так, чтобы промежуток был занят
  движением, без пауз в конце; потерявший линию робот едет дальше;
- «рука» везде с одной скоростью: время перестановки — от расстояния и угла (HAND_V, HAND_W), не от промежутка.
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
from manim import Circle, FadeOut, VGroup, config  # noqa: E402

import model  # noqa: E402
import nano_model  # noqa: E402
from nano_track import SENSOR  # noqa: E402
from params import Controller, Motor  # noqa: E402
from scenes.base import HAND, PLAYBACK, PIDScene, RunClock, RunSeries, render as _render  # noqa: E402,F401
from visuals.camera import Camera  # noqa: E402
from visuals.field import Field  # noqa: E402
from visuals.panel import GraphPanel  # noqa: E402
from visuals.robot import Robot  # noqa: E402

I_COLOR = "#F87171"          # И: красный — отличается от розового D, при дальтонизме теплее зеленого P
Z_DECOR = 0.2                # над линиями поля (0), под колесами (1)
SLACK = 0.25                 # запас до конца промежутка, с: округление кадров и пауза «все видно»
BLOCK = 10.0                 # длина промежутка сцены нового курса, с
HAND_V = 150.0               # «рука»: средняя скорость переноса, мм/с (время модели; на экране — с замедлением HAND)
HAND_W = 120.0               # «рука»: средняя скорость поворота, °/с
HAND_MIN = 0.4               # самая короткая перестановка, с


def hand_time(a, b):
    """Время перестановки «рукой» из позы a в позу b (сдвиг и поворот вместе), с — одна скорость для всех сцен."""
    dpsi = abs((b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi)
    return max(np.hypot(b[0] - a[0], b[1] - a[1]) / HAND_V, np.degrees(dpsi) / HAND_W, HAND_MIN)


def rotation(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


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

    robot_cls — класс робота (урок 4: с дальномером); field_items(view) — свои элементы поля (кубик, луч).
    """
    robot_cls = Robot

    def field_items(self, view):
        return []

    def setup(self):
        super().setup()
        self.sensor = SENSOR

    def end_block(self):
        """Конец промежутка BLOCK = 10 с: пауза до границы ровно в кадрах (как в PIDScene, но свой BLOCK).
        Предупреждения: промежуток длиннее BLOCK или в конце стоим без движения дольше 0,8 с."""
        self._blocks = getattr(self, "_blocks", 0) + 1
        need = round(self._blocks * BLOCK * config.frame_rate) - getattr(self, "frames", 0)
        if need < 0:
            warnings.warn(f"{type(self).__name__}: промежуток {self._blocks} длиннее BLOCK = {BLOCK} с "
                          f"на {-need / config.frame_rate:.1f} с")
        elif need / config.frame_rate > 0.8:
            warnings.warn(f"{type(self).__name__}: в промежутке {self._blocks} пауза {need / config.frame_rate:.1f} с")
        if need > 0:
            self.wait((need - 0.5) / config.frame_rate, frozen_frame=False)

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
    Длина прогона — сколько влезает в промежуток: прогон + докатывание после motors(0, 0) заканчиваются
    за SLACK до конца промежутка. Потерявший линию робот едет дальше (как настоящий). Между прогонами робота
    возвращают «рукой» (hand_to), кривые гаснут и подпись меняется в это же время.
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

    def fit_run(self, pose, ctrl, video):
        """Прогон, который вместе с докатыванием занимает video секунд видео (минус SLACK)."""
        target = (video - SLACK) * self.speed
        t_end = target - 0.25
        for _ in range(6):
            run = self.simulate(pose, ctrl, t_end)
            stop = model.brake(self.track, run, geom=self.geom, sensor=self.sensor, motor=Motor())
            over = run.t[-1] + stop.t[-1] - target
            if -0.01 <= over <= 0:
                break
            t_end -= over + 0.005
        return run

    def plan(self):
        """Прогоны всех промежутков заранее (шкала времени графика — по самому длинному). Время на возврат «рукой»
        зависит от того, где остановился прошлый прогон, — поэтому по очереди."""
        runs, video = [], BLOCK - 0.8                                   # первый: после заголовка
        for i, run in enumerate(self.runs):
            runs.append(self.fit_run(self.start(i), run[0], video))
            stop = model.brake(self.track, runs[-1], geom=self.geom, sensor=self.sensor, motor=Motor())
            end = (stop.x[-1], stop.y[-1], stop.psi[-1])
            if i + 1 < len(self.runs):
                video = BLOCK - hand_time(end, self.start(i + 1)) / HAND - self.t_on
        return runs

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
        for i, ((ctrl, caption, *text), run) in enumerate(zip(self.runs, runs)):   # прогон — промежуток BLOCK
            if i > 0:
                self.hand_to(self.start(i), FadeOut(curves), panel.set_text(*text, caption=caption))
                self.switch(run)
                self.play(readout.opacity.animate.set_value(1.0), run_time=self.t_on)
            clock = RunClock(self.t)
            curves = panel.curves(self.traj, clock)
            self.add(curves)
            self.advance(self.traj.t[-1])
            clock.stop()
            self.switch(model.brake(self.track, self.traj, geom=self.geom, sensor=self.sensor, motor=Motor()))
            self.advance(self.traj.t[-1], readout.opacity.animate.set_value(0.35))
            self.end_block()


def render(scene_cls, quality="480p", still=False, preview=True):
    """Как scenes.base.render: видео в videos/<Сцена>_<качество>.mp4 (общая папка с олимпиадными сценами)."""
    _render(scene_cls, quality, still, preview)
