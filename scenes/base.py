"""Общая основа сцен: макет, трекер времени t, привязка траектории к роботу и миру."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manim import Scene, ValueTracker, linear, tempconfig  # noqa: E402

import numpy as np  # noqa: E402

import model  # noqa: E402

from params import Geometry, Sensor, Style  # noqa: E402
from visuals.panels import Frame, backdrop  # noqa: E402
from visuals.robot import Robot  # noqa: E402
from visuals.world import World, XRay  # noqa: E402


class PIDScene(Scene):
    speed = 1.0   # скорость проигрывания: 0.5 — вдвое медленнее реального времени

    def setup(self):
        self.fr, self.st, self.geom, self.sensor = Frame(), Style(), Geometry(), Sensor()
        self.t = ValueTracker(0.0)
        self.add(backdrop(self.fr, self.st))

    def bind(self, traj, track, readout=None):
        """Робот и мир, которые следуют траектории self.traj по трекеру self.t; readout — индикатор ШИМ колёс."""
        self.traj = traj
        world = World(self.fr, track, st=self.st)
        robot = Robot(self.fr, self.geom, self.st, self.sensor)
        xray = XRay(self.fr, track, self.geom, st=self.st) if self.st.xray else None

        self.s_off = np.zeros(2)   # перенос пути колёс между отрезками, чтобы протектор не прыгал

        def follow(m):
            tr, t = self.traj, self.t.get_value()
            p = tr.at(t)
            self.wheel_s = np.array([p["sL"], p["sR"]]) + self.s_off
            robot.set_pose(p["psi"]).set_wheel_phase(*self.wheel_s)
            robot.set_signals(*self.levels(tr, t))
            world.set_robot_xy(p["x"], p["y"])
            if xray is not None:
                xray.set_pose(p["x"], p["y"], p["psi"])
            if readout is not None:
                sig = tr.signals_at(t)
                readout.set_values(p["psi"], sig["pwmL"], sig["pwmR"])

        robot.add_updater(follow)
        follow(robot)
        movers = [m for m in (world, xray, readout) if m is not None]
        for m in movers:
            # их двигает апдейтер робота; пустой апдейтер говорит Manim перерисовывать их каждый кадр
            m.add_updater(lambda m: None)
        self.add(robot, *movers)
        self.robot, self.world, self.readout = robot, world, readout
        return robot, world

    def switch(self, traj, t=0.0):
        """Подменить проигрываемую траекторию (новый прогон или переход); путь колёс продолжается."""
        p = traj.at(t)
        self.s_off = self.wheel_s - np.array([p["sL"], p["sR"]])
        self.traj = traj
        self.t.set_value(t)
        self.robot.update(0)

    def return_to(self, line, target, t_slide=1.0, t_turn=1.0):
        """Возврат между прогонами: линия подъезжает так, что середина оси встаёт в target[:2] (колёса не крутятся),
        затем робот поворачивается на месте на target[2]. Возвращает target."""
        x, y, psi = self.end_pose()
        tx, ty, tpsi = target
        if np.hypot(tx - x, ty - y) > 0.5:
            self.switch(model.move(line, (x, y, psi), (tx, ty, psi), t_slide, roll=False,
                                   geom=self.geom, sensor=self.sensor))
            self.advance(t_slide, speed=1.0)
        # кратчайший поворот к целевому курсу
        dpsi = (tpsi - psi + np.pi) % (2 * np.pi) - np.pi
        if abs(dpsi) > np.radians(0.5):
            self.switch(model.move(line, (tx, ty, psi), (tx, ty, psi + dpsi), t_turn,
                                   geom=self.geom, sensor=self.sensor))
            self.advance(t_turn, speed=1.0)
        return tx, ty, tpsi

    def end_pose(self):
        p = self.traj.at(self.t.get_value())
        return p["x"], p["y"], p["psi"]


    def levels(self, traj, t):
        """Показания физических датчиков [левый, правый] в момент t, нормированные: 0 — белое, 1 — чёрное."""
        sig = traj.signals_at(t)
        phys = np.array(model.to_channels(np.array([sig["L"], sig["R"]]), self.sensor))   # A0/A1 → левый/правый
        w, b = np.asarray(self.sensor.white, float), np.asarray(self.sensor.black, float)
        return (phys - w) / (b - w)

    def advance(self, t_to, *extra, speed=None):
        """Проиграть траекторию до момента t_to (плюс одновременные анимации extra)."""
        dur = (t_to - self.t.get_value()) / (speed or self.speed)
        self.play(self.t.animate.set_value(t_to), *extra, run_time=dur, rate_func=linear)


def render(scene_cls, quality="low_quality", still=False, preview=True):
    """Запуск кнопкой ▶ в PyCharm.

    Видео → media/videos/<Сцена>/, still=True — последний кадр png → renders/<Сцена>.png.
    quality: "low_quality" (480p, черновик), "medium_quality" (720p), "high_quality" (1080p).
    То же, что `manim -pql scenes/<файл>.py <Сцена>` (с -s для кадра).
    """
    name = scene_cls.__name__
    cfg = {"quality": quality, "preview": preview, "output_file": name,
           "media_dir": str(ROOT / "media"), "video_dir": str(ROOT / "media" / "videos" / name)}
    if still:
        cfg.update(save_last_frame=True, write_to_movie=False, images_dir=str(ROOT / "renders"))
    with tempconfig(cfg):
        scene_cls().render()
