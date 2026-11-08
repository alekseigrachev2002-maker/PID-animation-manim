"""Основа сцен.

PIDScene — стенд: трекер времени t, камера, поле, робот и элементы, которым каждый кадр раздаётся состояние.
RunSeries — серия прогонов регулятора: прогон → стоп → возврат → следующий (сцены 2a, 2b и дальше).
render() — запуск кнопкой ▶ в PyCharm.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from manim import FadeOut, Scene, ValueTracker, linear, tempconfig  # noqa: E402

import model  # noqa: E402
from params import Controller, Geometry, Motor, Sensor, Style  # noqa: E402
from visuals.camera import Camera  # noqa: E402
from visuals.field import Field  # noqa: E402
from visuals.layout import Frame, backdrop  # noqa: E402
from visuals.panel import ERR, GraphPanel  # noqa: E402
from visuals.robot import Robot  # noqa: E402


class PIDScene(Scene):
    speed = 1.0   # скорость проигрывания: 0.5 — вдвое медленнее реального времени

    def setup(self):
        self.fr, self.st, self.geom, self.sensor = Frame(), Style(), Geometry(), Sensor()
        self.t = ValueTracker(0.0)
        self.add(backdrop(self.fr, self.st))

    # ---------- стенд ----------

    def show(self, track, traj, camera="follow", center=(0.0, 0.0), sensors=True, items=()):
        """Поставить на стенд поле, робота и элементы items (у каждого — update_state(state))."""
        self.track, self.traj = track, traj
        self.view = Camera(self.fr, camera, center)
        self.field = Field(self.view, track, self.sensor, self.st)
        self.robot = Robot(self.fr, self.geom, self.st, self.sensor, sensors)
        self.items = [m for m in items if m is not None]
        self.s_off = np.zeros(2)   # перенос пути колёс между отрезками, чтобы протектор не прыгал

        def follow(_):
            s = self.state()
            self.view.follow(s["x"], s["y"])
            self.field.update_view()
            self.robot.place(self.view.point(s["x"], s["y"]), s["psi"])
            self.robot.set_wheel_phase(s["sL"], s["sR"]).set_signals(*s["levels"])
            for m in self.items:
                m.update_state(s)

        self.robot.add_updater(follow)
        follow(self.robot)
        for m in (self.field, *self.items):
            # их двигает апдейтер робота; пустой апдейтер говорит Manim перерисовывать их каждый кадр
            m.add_updater(lambda m: None)
        self.add(self.field, self.robot, *self.items)

    def state(self):
        """Состояние в момент t: поза, путь колёс, сигналы регулятора, яркость меток датчиков."""
        t = self.t.get_value()
        s = {**self.traj.at(t), **self.traj.signals_at(t)}
        s["sL"], s["sR"] = np.array([s["sL"], s["sR"]]) + self.s_off
        self.wheel_s = np.array([s["sL"], s["sR"]])
        phys = np.array(model.to_channels(np.array([s["L"], s["R"]]), self.sensor))   # A0/A1 → левый/правый
        w, b = np.asarray(self.sensor.white, float), np.asarray(self.sensor.black, float)
        s["levels"] = (phys - w) / (b - w)
        return s

    def switch(self, traj, t=0.0):
        """Подменить проигрываемую траекторию (новый прогон или переход); путь колёс продолжается."""
        p = traj.at(t)
        self.s_off = self.wheel_s - np.array([p["sL"], p["sR"]])
        self.traj = traj
        self.t.set_value(t)
        self.robot.update(0)

    def end_pose(self):
        p = self.traj.at(self.t.get_value())
        return p["x"], p["y"], p["psi"]

    def advance(self, t_to, *extra, speed=None):
        """Проиграть траекторию до момента t_to (плюс одновременные анимации extra)."""
        dur = (t_to - self.t.get_value()) / (speed or self.speed)
        self.play(self.t.animate.set_value(t_to), *extra, run_time=dur, rate_func=linear)

    def return_to(self, target, t_slide=1.0, t_turn=1.0):
        """Возврат между прогонами: линия подъезжает так, что середина оси встаёт в target[:2] (колёса не крутятся),
        затем робот поворачивается на месте на курс target[2]."""
        x, y, psi = self.end_pose()
        tx, ty, tpsi = target
        if np.hypot(tx - x, ty - y) > 0.5:
            self.switch(model.move(self.track, (x, y, psi), (tx, ty, psi), t_slide, roll=False,
                                   geom=self.geom, sensor=self.sensor))
            self.advance(t_slide, speed=1.0)
        dpsi = (tpsi - psi + np.pi) % (2 * np.pi) - np.pi      # кратчайший поворот
        if abs(dpsi) > np.radians(0.5):
            self.switch(model.move(self.track, (tx, ty, psi), (tx, ty, psi + dpsi), t_turn,
                                   geom=self.geom, sensor=self.sensor))
            self.advance(t_turn, speed=1.0)


class RunSeries(PIDScene):
    """Серия прогонов регулятора по одной трассе. Сцена задаёт трассу, старт и прогоны:

        runs = [(0.3, "kp = 0.3, едет по линии"), (dict(kp=0.3, kd=2), "с D")]   # kp или параметры регулятора
        def make_track(self): ...                 # трасса
        def start(self, i): ...                   # поза старта прогона i (для i > 0 — после возврата)
    """
    phrase = "Подбираем P"
    formula = "control = kp·err"
    formula_colors = [("err", "e")]     # части формулы цветом своих кривых
    law = "kp·err"                     # слагаемое в формулах колёс: 150 ± law
    runs = []
    signals = [ERR]
    y = ("err", -1023, 1023)
    t_run = 2.0                        # длительность прогона, с
    t_ticks = [0, 0.5, 1, 1.5, 2]
    speed = 0.33                       # прогоны — втрое медленнее
    delay = 0.020                      # задержка команда → колесо, с (заглушка до замеров)
    t_slide = 1.0                      # возврат: линия подъезжает, с
    t_turn = 1.0                       # возврат: поворот на месте, с

    def make_track(self):
        raise NotImplementedError

    def start(self, i):
        raise NotImplementedError

    def simulate(self, pose, ctrl):
        ctrl = Controller(**({"kp": ctrl} if np.isscalar(ctrl) else ctrl))
        return model.cached_simulate(self.track, tuple(float(v) for v in pose), ctrl=ctrl,
                                     motor=Motor(delay=self.delay), t_end=self.t_run)

    def construct(self):
        self.track = self.make_track()
        traj = self.simulate(self.start(0), self.runs[0][0])
        panel = GraphPanel(self.fr, self.st, self.phrase, self.runs[0][1], self.y, self.t_run * 1.12,
                           self.t_ticks, self.signals,
                           readout=dict(signs=model.wheel_signs(Motor()), law=self.law))
        self.show(self.track, traj, items=[panel.readout])
        self.add(panel)
        self.play(panel.set_formula(self.formula, self.formula_colors), run_time=0.8)
        readout = panel.readout
        for i, (ctrl, caption) in enumerate(self.runs):
            if i > 0:
                target = self.start(i)
                self.return_to(target, self.t_slide, self.t_turn)
                self.switch(self.simulate(target, ctrl))
                self.play(readout.opacity.animate.set_value(1.0), run_time=0.4)
            clock = RunClock(self.t)            # кривые растут за t во время прогона и замирают на его конце
            curves = panel.curves(self.traj, clock)
            self.add(curves)
            self.wait(0.3)
            self.advance(self.traj.t[-1])
            clock.stop()
            # стоп: motors(0, 0), робот докатывается по той же физике; строка ШИМ приглушается — регулятор выключен
            self.switch(model.brake(self.track, self.traj, geom=self.geom, sensor=self.sensor,
                                    motor=Motor(delay=self.delay)))
            self.advance(self.traj.t[-1], readout.opacity.animate.set_value(0.35))
            self.wait(0.5)
            if i < len(self.runs) - 1:
                # кривые гаснут, в шапке — следующий прогон
                self.play(FadeOut(curves), panel.set_caption(self.runs[i + 1][1]), run_time=0.8)
        self.wait(1.0)


class RunClock:
    """Время для кривых прогона: идёт за трекером t, после stop() замирает (дальше t перезапускается)."""

    def __init__(self, t):
        self.t, self.live, self.value = t, True, 0.0

    def stop(self):
        self.live = False

    def __call__(self):
        if self.live:
            self.value = self.t.get_value()
        return self.value


def render(scene_cls, quality="low_quality", still=False, preview=True):
    """Запуск кнопкой ▶ в PyCharm.

    Видео → media/videos/<Сцена>/, still=True — последний кадр png → renders/<Сцена>.png.
    quality: "low_quality" (480p, черновик), "medium_quality" (720p), "high_quality" (1080p).
    """
    name = scene_cls.__name__
    cfg = {"quality": quality, "preview": preview, "output_file": name,
           "media_dir": str(ROOT / "media"), "video_dir": str(ROOT / "media" / "videos" / name)}
    if still:
        cfg.update(save_last_frame=True, write_to_movie=False, images_dir=str(ROOT / "renders"))
    with tempconfig(cfg):
        scene_cls().render()
