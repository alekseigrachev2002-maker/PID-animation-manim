"""Сцена 2a. Подбор P: серия прогонов из одной позы, прошлые кривые — бледные призраки."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import ValueTracker, smooth  # noqa: E402

import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Motor  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from visuals.panels import Header, Plot, legend  # noqa: E402

U_P = "u = kp·e"

# (kp, пояснение в шапке) — в порядке показа
RUNS = [
    (0.0, "kp = 0, робот не поворачивает"),
    (0.1, "kp = 0.1, возвращается медленно"),
    (0.5, "kp = 0.5, перехлёсты"),
    (0.8, "kp = 0.8, раскачка"),
    (0.3, "kp = 0.3, быстро и без раскачки"),
]


class Scene2a(PIDScene):
    psi0_deg = 15.0     # стартовый поворот относительно линии
    delay = 0.020       # задержка команда → колесо, с (заглушка до замеров)
    t_run = 2.0         # длительность прогона, с
    speed = 0.33        # проигрывание втрое медленнее
    ghost = 0.22        # непрозрачность прошлых прогонов

    def trajectories(self, line):
        pose0 = (0.0, 0.0, np.radians(self.psi0_deg))
        return [model.cached_simulate(line, pose0, ctrl=Controller(kp=kp), motor=Motor(delay=self.delay),
                                      t_end=self.t_run) for kp, _ in RUNS]

    def curve(self, plot, traj):
        """Кривая e прогона: растёт за трекером t и не укорачивается при перемотке."""
        reached = [0.0]

        def end():
            reached[0] = max(reached[0], self.t.get_value())
            return reached[0]

        op = ValueTracker(1.0)
        return plot.curve(traj.tt, traj.e, self.st.e, end, op, width_px=3), op

    def construct(self):
        fr, st = self.fr, self.st
        line = track.straight(y1=6000)
        trajs = self.trajectories(line)
        self.bind(trajs[0], line, pwm=True)

        plot = Plot(fr, t_max=self.t_run, lo=-1023, hi=1023, st=st)
        ticks = {v: (lambda lo, hi: 1) for v in (-1023, 0, 1023)}
        header = Header(fr, "Подбираем P", RUNS[0][1], st)
        leg = legend(fr, [(st.e, "e = L − R", 1), (st.e, "прошлые прогоны", self.ghost)], st)
        self.add(plot.axes(ticks, y_label="e", t_ticks=[0, 0.5, 1, 1.5, 2]), header, leg)
        self.play(header.set_formula(U_P, st.e), run_time=0.8)

        for i, (traj, (kp, caption)) in enumerate(zip(trajs, RUNS)):
            if i > 0:
                self.switch(traj)
                self.play(header.set_caption(caption), run_time=0.6)
            c, op = self.curve(plot, traj)
            self.add(c)
            self.wait(0.3)
            t_end = self.t_run if np.isnan(traj.lost_at) else min(self.t_run, traj.lost_at + 0.3)
            self.advance(t_end)
            self.wait(0.6)
            if i < len(RUNS) - 1:
                # прогон становится призраком, робот отматывается к старту
                self.play(op.animate.set_value(self.ghost),
                          self.t.animate.set_value(0.0), run_time=1.0, rate_func=smooth)
        self.wait(1.5)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p: render(Scene2a, "high_quality", still=True)
    render(Scene2a)
