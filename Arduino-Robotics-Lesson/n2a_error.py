"""Урок 2, сцена 2a. Ошибка err = L − R: робота качают «рукой» на месте над верхней внутренней прямой трека
(камера повернута: робот смотрит вверх, по ходу — по часовой стрелке).

Как олимпиадная сцена 1: подсцена 1 — L и R (0…1023); подсцена 2 — то же качание, L и R бледнеют,
шкала ±1023, под ними рисуется err. Линия 20 мм: пока она между датчиками, err ≈ 0 — «мертвая зона» шире,
чем кажется; датчик целиком на линии — err ≈ ±700.
"""
from nano_base import HAND, NanoScene, render  # первым: добавляет в путь корень репозитория

import numpy as np
from manim import ValueTracker

import model
from nano_track import ShortTrack
from scenes.s1_error import swing
from visuals.panel import ERR, GraphPanel, Signal

PLACE = 2400.0      # путь по маршруту (по часовой), мм: верхняя внутренняя прямая
T_SWAP = 1.2        # смена шкалы и легенды перед вторым проходом, с


class Nano2a(NanoScene):
    speed = HAND
    amp_deg = 15.0
    period = 1.7       # период качания, с — как в олимпиадной сцене 1 (рука с той же скоростью)
    cycles = 3
    ramp = 0.3

    def construct(self):
        st, sens = self.st, self.sensor
        track = ShortTrack()
        x, y, psi0 = track.pose_at(PLACE)
        t, psi = swing(self.amp_deg, self.period, self.cycles, self.ramp)
        one = model.prescribed(track, t, x, y, psi0 + psi, sensor=sens)
        T = one.t[-1]
        self.show(track, model.chain(one, one))

        a0, a1 = model.channel_sides(sens)
        L, R = Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")
        prog = ValueTracker(0.0)
        ticks = {0: lambda lo, hi: 1, 512: lambda lo, hi: 1 - prog.get_value(), 1023: lambda lo, hi: 1,
                 -1023: lambda lo, hi: prog.get_value()}
        panel = GraphPanel(self.fr, st, "Считаем ошибку", "сигналы левого и правого датчиков",
                           y=("АЦП", 0, 1023), t_max=T * 1.06, t_ticks=range(int(T) + 1), signals=[L, R], ticks=ticks)
        ref_op = ValueTracker(1.0)
        refs = [panel.plot.hline(sens.black[0], f"черный ≈ {sens.black[0]:.0f}", ref_op),
                panel.plot.hline(sens.white[0], f"белый ≈ {sens.white[0]:.0f}", ref_op)]
        dim = ValueTracker(1.0)
        cL, cR = panel.curves(one, lambda: min(self.t.get_value(), T), dim)
        self.add(panel, *refs, cL, cR)
        self.play(panel.intro(), run_time=0.8)

        self.begin_segment()
        self.advance(T)
        self.end_segment(check=False)                     # качание «рукой»: длина — от периода, не 8 с

        self.play(dim.animate.set_value(0.35), prog.animate.set_value(1.0), panel.plot.lo.animate.set_value(-1023),
                  ref_op.animate.set_value(0.0),
                  panel.set_legend([(Signal("L", "L", "L"), 0.35), (Signal("R", "R", "R"), 0.35), ERR]),
                  panel.set_text(ERR.label, [("err", "e"), ("L", "L"), ("R", "R")], "разность сигналов"),
                  run_time=T_SWAP)
        self.add(panel.curve(one, ERR, lambda: max(self.t.get_value() - T, 0.0)))
        self.begin_segment()
        full = round(abs(sens.white[0] - sens.black[0]), -2)
        t_full = float(one.tt[np.argmax(np.abs(one.e) > 0.9 * full)])
        self.advance(T + t_full)
        self.advance(T + t_full + 0.8 * self.speed, panel.set_caption(f"err ≈ ±{full:.0f}, датчик целиком на линии"))
        self.advance(2 * T)
        self.end_segment(check=False)


if __name__ == "__main__":
    render(Nano2a, "480p")
