"""Сцена 1. Качание на месте над прямой: L и R → то же качание еще раз, под ними рисуется err = L − R."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import ValueTracker  # noqa: E402

import model  # noqa: E402
from scenes.base import HAND, PIDScene, render  # noqa: E402
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
    """Два промежутка, в каждом — одно и то же качание «рукой» (3 размаха) на весь график.

    1: L и R. 2: L и R бледнеют и остаются, шкала ±1023; робота качают так же, и err рисуется под ними — каждая
    точка err ровно там, где ее L и R (то же качание, та же ось времени).
    """
    speed = HAND       # качание «рукой» — как остальные движения рукой (езды здесь нет, PLAYBACK не нужен)
    amp_deg = 15.0     # амплитуда качания
    period = 1.7       # период качания, с: три размаха с переходом влезают в промежуток BLOCK
    cycles = 3         # размахов за проход
    ramp = 0.3         # плавный старт и остановка качания, с

    def construct(self):
        st, sens = self.st, self.sensor
        track = Track(Route(start=(0, -300), heading="вверх").forward(3300))
        t, psi = swing(self.amp_deg, self.period, self.cycles, self.ramp)
        one = model.prescribed(track, t, 0.0, 0.0, psi, sensor=sens)     # один проход: начинается и кончается в покое
        T = one.t[-1]
        self.show(track, model.chain(one, one))                         # два одинаковых прохода подряд

        a0, a1 = model.channel_sides(sens)
        L, R = Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")
        prog = ValueTracker(0.0)   # 0: шкала 0…1023, 1: шкала ±1023
        ticks = {0: lambda lo, hi: 1, 512: lambda lo, hi: 1 - prog.get_value(), 1023: lambda lo, hi: 1,
                 -1023: lambda lo, hi: prog.get_value()}
        panel = GraphPanel(self.fr, st, "Считаем ошибку", "сигналы левого и правого датчиков",
                           y=("АЦП", 0, 1023), t_max=T * 1.06, t_ticks=range(int(T) + 1), signals=[L, R], ticks=ticks)
        ref_op = ValueTracker(1.0)
        refs = [panel.plot.hline(sens.black[0], f"черный ≈ {round(sens.black[0], -2):.0f}", ref_op),   # подписи — круглые,
                panel.plot.hline(sens.white[0], f"белый ≈ {round(sens.white[0], -2):.0f}", ref_op)]    # справа: слева кривые
        dim = ValueTracker(1.0)
        cL, cR = panel.curves(one, lambda: min(self.t.get_value(), T), dim)   # L и R — по первому проходу
        self.add(panel, *refs, cL, cR)
        self.play(panel.intro(), run_time=0.8)          # заголовок выезжает слева

        # промежуток 1: качание — L и R на весь график
        self.advance(T)
        self.end_block()

        # промежуток 2: L, R бледнеют и остаются, шкала ±1023, появляется err; то же качание — err рисуется под ними
        self.play(dim.animate.set_value(0.35), prog.animate.set_value(1.0), panel.plot.lo.animate.set_value(-1023),
                  ref_op.animate.set_value(0.0),
                  panel.set_legend([(Signal("L", "L", "L"), 0.35), (Signal("R", "R", "R"), 0.35), ERR]),
                  panel.set_text(ERR.label, [("err", "e"), ("L", "L"), ("R", "R")], "разность сигналов"),
                  run_time=2.0)                         # смена шкалы и легенды — не спеша
        self.add(panel.curve(one, ERR, lambda: max(self.t.get_value() - T, 0.0)))   # err второго прохода — на месте первого
        full = round(abs(sens.white[0] - sens.black[0]), -2)        # |err|, когда датчик целиком на линии
        t_full = float(one.tt[np.argmax(np.abs(one.e) > 0.9 * full)])   # err впервые выходит на плато
        self.advance(T + t_full)
        self.advance(T + t_full + 0.8 * self.speed, panel.set_caption(f"err ≈ ±{full:.0f}, датчик целиком на линии"))
        self.advance(2 * T)
        self.end_block()


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: видео в videos/Scene1_<качество>.mp4. Качество: "240p", "480p", "720p", "1080p", "4k".
    render(Scene1, "480p")
    # render(Scene1, "1080p")                  # для презентации; все шесть сразу — scenes/lessons.py
