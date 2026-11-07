"""Сцена 2a. Подбор P: три прогона из одной позы — недорегулирование, хорошо, перерегулирование."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import FadeOut  # noqa: E402

import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Motor  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from visuals.panels import Header, Plot, legend  # noqa: E402

U_P = "u = kp·e"

# (kp, пояснение в шапке) — в порядке показа; сцена заканчивается раскачкой, 2b спускается на «хорошо» и вводит D
RUNS = [
    (0.1, "kp = 0.1, недорегулирование"),
    (0.3, "kp = 0.3, хорошо"),
    (0.8, "kp = 0.8, перерегулирование"),
]


class Scene2a(PIDScene):
    psi0_deg = 15.0     # стартовый поворот относительно линии
    delay = 0.020       # задержка команда → колесо, с (заглушка до замеров)
    t_run = 2.0         # длительность прогона, с
    speed = 0.33        # проигрывание втрое медленнее

    def trajectories(self, line):
        pose0 = (0.0, 0.0, np.radians(self.psi0_deg))
        return [model.cached_simulate(line, pose0, ctrl=Controller(kp=kp), motor=Motor(delay=self.delay),
                                      t_end=self.t_run) for kp, _ in RUNS]

    def construct(self):
        fr, st = self.fr, self.st
        line = track.straight(y1=6000)
        trajs = self.trajectories(line)
        self.bind(trajs[0], line, pwm=True)

        plot = Plot(fr, t_max=self.t_run, lo=-1023, hi=1023, st=st)
        ticks = {v: (lambda lo, hi: 1) for v in (-1023, 0, 1023)}
        header = Header(fr, "Подбираем P", RUNS[0][1], st)
        leg = legend(fr, [(st.e, "e = L − R", 1)], st)
        self.add(plot.axes(ticks, y_label="e", t_ticks=[0, 0.5, 1, 1.5, 2]), header, leg)
        self.play(header.set_formula(U_P, st.e), run_time=0.8)

        curve = None
        for i, (traj, (kp, caption)) in enumerate(zip(trajs, RUNS)):
            if i > 0:
                # кривая гаснет, робота «переставляют руками» на старт — без перемотки траектории
                self.play(FadeOut(curve), self.hide_robot(), header.set_caption(caption), run_time=0.8)
                self.switch(traj)
                self.play(self.show_robot(), run_time=0.5)
            curve = plot.curve(traj.tt, traj.e, st.e, self.t, width_px=3)
            self.add(curve)
            self.wait(0.3)
            self.advance(self.t_run)
            self.wait(0.8)
        self.wait(1.0)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p: render(Scene2a, "high_quality", still=True)
    render(Scene2a)
