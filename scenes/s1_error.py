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


def swing(amp_deg, period, cycles, dt=0.005):
    """Качание ψ(t) = A·sin(2πt/T): сначала влево."""
    t = np.arange(0, cycles * period + dt / 2, dt)
    return t, np.radians(amp_deg) * np.sin(2 * np.pi * t / period)


class Scene1(PIDScene):
    amp_deg = 15.0     # амплитуда качания
    period = 2.0       # период качания, с
    speed = 0.5        # проигрывание вдвое медленнее

    def construct(self):
        st, sens = self.st, self.sensor
        track = Track(Route(start=(0, -300), heading="вверх").forward(3300))
        t, psi = swing(self.amp_deg, self.period, 2)
        self.show(track, model.prescribed(track, t, 0.0, 0.0, psi, sensor=sens))

        a0, a1 = model.channel_sides(sens)
        L, R = Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")
        prog = ValueTracker(0.0)   # 0: шкала 0…1023, 1: шкала ±1023
        ticks = {0: lambda lo, hi: 1, 512: lambda lo, hi: 1 - prog.get_value(), 1023: lambda lo, hi: 1,
                 -1023: lambda lo, hi: prog.get_value()}
        panel = GraphPanel(self.fr, st, "Считаем ошибку", "сигналы левого и правого датчиков",
                           y=("АЦП", 0, 1023), t_max=4.8, t_ticks=range(5), signals=[L, R], ticks=ticks)
        ref_op = ValueTracker(1.0)
        refs = [panel.plot.hline(sens.black[0], f"чёрный ≈ {sens.black[0]}", ref_op),
                panel.plot.hline(sens.white[0], f"белый ≈ {sens.white[0]}", ref_op)]
        dim = ValueTracker(1.0)
        cL, cR = panel.curve(self.traj, L, self.t, dim), panel.curve(self.traj, R, self.t, dim)
        self.add(panel, *refs, cL, cR)
        self.wait(0.5)

        # 1. одно качание: L и R
        self.advance(self.period)

        # 2. L, R бледнеют, шкала ±1023, появляется err
        e_end = ValueTracker(0.0)
        cE = panel.curve(self.traj, Signal(ERR.label, "e", "e", 3), lambda: min(e_end.get_value(), self.t.get_value()))
        self.play(dim.animate.set_value(0.35), prog.animate.set_value(1.0), panel.plot.lo.animate.set_value(-1023),
                  ref_op.animate.set_value(0.0),
                  panel.set_legend([(Signal("L", "L", "L"), 0.35), (Signal("R", "R", "R"), 0.35), ERR]),
                  panel.set_formula(ERR.label), panel.set_caption("разность сигналов"), run_time=1.2)
        self.add(cE)
        self.play(e_end.animate.set_value(self.t.get_value()), run_time=1.5, rate_func=linear)
        e_end.set_value(1e9)

        # 3. ещё одно качание
        self.play(panel.set_caption("err = 0, пока оба на белом"), run_time=0.8)
        self.advance(2 * self.period)
        self.wait(1.5)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p:
    render(Scene1, "high_quality", still=True)
    # render(Scene1, "high_quality")
