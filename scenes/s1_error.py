"""Сцена 1. Поворот на месте над прямой: L и R → появляется e = L − R, видно плато e = 0."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import FadeIn, FadeOut, ValueTracker, linear  # noqa: E402

import model  # noqa: E402
import track  # noqa: E402
from params import Sensor  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from visuals.panels import Plot, Header, legend, text  # noqa: E402

E = "<i>e</i> = <i>L</i> − <i>R</i>"


def swing(amp_deg, period, cycles, dt=0.005):
    """Качание ψ(t) = A·sin(2πt/T): сначала влево."""
    t = np.arange(0, cycles * period + dt / 2, dt)
    return t, np.radians(amp_deg) * np.sin(2 * np.pi * t / period)


class Scene1(PIDScene):
    amp_deg = 18.0     # амплитуда качания
    period = 2.0       # период качания, с
    speed = 0.5        # проигрывание вдвое медленнее
    captions = False   # подписи-субтитры

    def construct(self):
        fr, st = self.fr, self.st
        line = track.straight()
        t, psi = swing(self.amp_deg, self.period, 2)
        traj = model.prescribed(line, t, 0.0, 0.0, psi)
        self.bind(traj, line)

        plot = Plot(fr, t_max=4.8, lo=0, hi=1023, st=st)
        prog = ValueTracker(0.0)   # 0: шкала 0…1023, 1: шкала ±1023
        ticks = {0: lambda lo, hi: 1, 512: lambda lo, hi: 1 - prog.get_value(), 1023: lambda lo, hi: 1,
                 -1023: lambda lo, hi: prog.get_value()}
        ref_op = ValueTracker(1.0)
        sens = Sensor()
        refs = [plot.hline(sens.black[0], f"чёрный ≈ {sens.black[0]}", ref_op),
                plot.hline(sens.white[0], f"белый ≈ {sens.white[0]}", ref_op)]
        dim = ValueTracker(1.0)
        cL = plot.curve(traj.tt, traj.L, st.L, self.t, dim)
        cR = plot.curve(traj.tt, traj.R, st.R, self.t, dim)
        header = Header(fr, "Считаем ошибку", st)
        leg1 = legend(fr, [(st.L, "L — левый датчик, A0", 1), (st.R, "R — правый датчик, A1", 1)], st)
        self.add(plot.axes(ticks), *refs, header, leg1, cL, cR)
        self.wait(0.5)

        # 1. одно качание: L и R
        self.advance(self.period)

        # 2. L, R бледнеют, шкала ±1023, появляется e
        e_end = ValueTracker(0.0)
        cE = plot.curve(traj.tt, traj.e, st.e, lambda: min(e_end.get_value(), self.t.get_value()), width_px=3)
        leg2 = legend(fr, [(st.L, "L", 0.35), (st.R, "R", 0.35), (st.e, "e = L − R", 1)], st)
        self.play(dim.animate.set_value(0.35), prog.animate.set_value(1.0), plot.lo.animate.set_value(-1023),
                  ref_op.animate.set_value(0.0), FadeOut(leg1), FadeIn(leg2), header.set_formula(E, st.e),
                  run_time=1.2)
        self.add(cE)
        self.play(e_end.animate.set_value(self.t.get_value()), run_time=1.5, rate_func=linear)
        e_end.set_value(1e9)

        # 3. ещё одно качание
        self.advance(2 * self.period)
        if self.captions:
            cap = text(fr, "плато e = 0: оба датчика на белом", 15, st.e, st)
            cap.move_to(plot.to_screen([0.1], [-900])[0], aligned_edge=cap.get_left() - cap.get_center())
            self.play(FadeIn(cap.set_z_index(13)))
        self.wait(1.5)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p: render(Scene1, "high_quality", still=True)
    render(Scene1)
