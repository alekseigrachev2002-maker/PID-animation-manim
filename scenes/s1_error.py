"""Сцена 1. Поворот на месте над прямой: L и R → появляется err = L − R, видно плато err = 0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import ValueTracker, linear  # noqa: E402

import model  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from track import Route, Track  # noqa: E402
from visuals.panel import ERR, GraphPanel, Signal  # noqa: E402


def swing(amp_deg, period, cycles, ramp=0.3, dt=0.005):
    """Качание ψ = A·sin(2π·u/T), начинается и кончается в покое.

    «Внутреннее время» u разгоняется от 0 до темпа часов за ramp секунд и так же тормозит в конце, поэтому
    все размахи полные, а скорость поворота в начале и в конце нулевая. Длительность — cycles·T + ramp.
    """
    T_u = cycles * period
    T = T_u + ramp
    t = np.arange(0, T + dt / 2, dt)
    u = np.where(t < ramp, t ** 2 / (2 * ramp),
                 np.where(t > T - ramp, T_u - (T - t) ** 2 / (2 * ramp), t - ramp / 2))
    return t, np.radians(amp_deg) * np.sin(2 * np.pi * u / period)


class Scene1(PIDScene):
    amp_deg = 15.0     # амплитуда качания
    period = 2.0       # период качания, с
    ramp = 0.3         # плавный старт и остановка качания, с

    def construct(self):
        st, sens = self.st, self.sensor
        track = Track(Route(start=(0, -300), heading="вверх").forward(3300))
        t, psi = swing(self.amp_deg, self.period, 2, self.ramp)
        self.show(track, model.prescribed(track, t, 0.0, 0.0, psi, sensor=sens))

        a0, a1 = model.channel_sides(sens)
        L, R = Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")
        prog = ValueTracker(0.0)   # 0: шкала 0…1023, 1: шкала ±1023
        ticks = {0: lambda lo, hi: 1, 512: lambda lo, hi: 1 - prog.get_value(), 1023: lambda lo, hi: 1,
                 -1023: lambda lo, hi: prog.get_value()}
        panel = GraphPanel(self.fr, st, "Считаем ошибку", "сигналы левого и правого датчиков",
                           y=("АЦП", 0, 1023), t_max=4.8, t_ticks=range(5), signals=[L, R], ticks=ticks)
        ref_op = ValueTracker(1.0)
        refs = [panel.plot.hline(sens.black[0], f"черный ≈ {round(sens.black[0], -2):.0f}", ref_op),   # подписи — круглые,
                panel.plot.hline(sens.white[0], f"белый ≈ {round(sens.white[0], -2):.0f}", ref_op)]    # справа: слева кривые
        dim = ValueTracker(1.0)
        cL, cR = panel.curves(self.traj, self.t, dim)
        self.add(panel, *refs, cL, cR)
        self.play(panel.intro(), run_time=0.8)          # заголовок выезжает слева
        self.wait(0.5)

        # 1. одно качание: L и R
        self.advance(self.period + self.ramp / 2)      # конец первого качания

        # 2. L, R бледнеют, шкала ±1023, появляется err
        e_end = ValueTracker(0.0)
        cE = panel.curve(self.traj, ERR, lambda: min(e_end.get_value(), self.t.get_value()))
        self.play(dim.animate.set_value(0.35), prog.animate.set_value(1.0), panel.plot.lo.animate.set_value(-1023),
                  ref_op.animate.set_value(0.0),
                  panel.set_legend([(Signal("L", "L", "L"), 0.35), (Signal("R", "R", "R"), 0.35), ERR]),
                  panel.set_text(ERR.label, [("err", "e"), ("L", "L"), ("R", "R")], "разность сигналов"), run_time=1.2)
        self.add(cE)
        self.play(e_end.animate.set_value(self.t.get_value()), run_time=1.5, rate_func=linear)
        e_end.set_value(1e9)

        # 3. еще одно качание
        full = round(abs(sens.white[0] - sens.black[0]), -2)        # |err|, когда датчик целиком на линии
        self.play(panel.set_caption(f"err ≈ ±{full:.0f}, датчик целиком на линии"), run_time=0.8)
        self.advance(self.traj.t[-1])
        self.wait(1.5)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: видео в videos/Scene1_<качество>.mp4. Качество: "240p", "480p", "720p", "1080p", "4k".
    render(Scene1, "480p")
    # render(Scene1, "1080p")                  # для презентации; все шесть сразу — scenes/lessons.py
