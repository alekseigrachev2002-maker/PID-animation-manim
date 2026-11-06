"""Общая основа сцен: макет, трекер времени t, привязка траектории к роботу и миру."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manim import Scene, ValueTracker, linear, tempconfig  # noqa: E402

from params import Geometry, Style  # noqa: E402
from visuals.panels import Frame, backdrop  # noqa: E402
from visuals.robot import Robot  # noqa: E402
from visuals.world import World, XRay  # noqa: E402


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
        xray = XRay(self.fr, track, self.geom, st=self.st) if self.st.xray else None

        def follow(m):
            p = traj.at(self.t.get_value())
            robot.set_pose(p["psi"]).set_wheel_phase(p["sL"], p["sR"])
            world.set_robot_xy(p["x"], p["y"])
            if xray is not None:
                xray.set_pose(p["x"], p["y"], p["psi"])

        robot.add_updater(follow)
        follow(robot)
        self.add(world, robot, *([xray] if xray else []))
        return robot, world

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
