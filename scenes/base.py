"""Общая основа сцен: макет, трекер времени t, привязка траектории к роботу и миру."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manim import Scene, ValueTracker, linear  # noqa: E402

from params import Geometry, Style  # noqa: E402
from visuals.panels import Frame, backdrop  # noqa: E402
from visuals.robot import Robot  # noqa: E402
from visuals.world import World  # noqa: E402


class PIDScene(Scene):
    speed = 1.0   # скорость проигрывания: 0.5 — вдвое медленнее реального времени

    def setup(self):
        self.fr, self.st, self.geom = Frame(), Style(), Geometry()
        self.t = ValueTracker(0.0)
        self.add(backdrop(self.fr, self.st))

    def bind(self, traj, track):
        """Робот и мир, которые следуют траектории traj по трекеру self.t."""
        world = World(self.fr, track, st=self.st)
        robot = Robot(self.fr, self.geom, self.st)

        def follow(m):
            p = traj.at(self.t.get_value())
            robot.set_pose(p["psi"]).set_wheel_phase(p["sL"], p["sR"])
            world.set_robot_xy(p["x"], p["y"])

        robot.add_updater(follow)
        follow(robot)
        self.add(world, robot)
        return robot, world

    def advance(self, t_to, *extra, speed=None):
        """Проиграть траекторию до момента t_to (плюс одновременные анимации extra)."""
        dur = (t_to - self.t.get_value()) / (speed or self.speed)
        self.play(self.t.animate.set_value(t_to), *extra, run_time=dur, rate_func=linear)
