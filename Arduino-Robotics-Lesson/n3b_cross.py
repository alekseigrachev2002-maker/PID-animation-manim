"""Урок 3, сцена 3b. Перекресток: оба датчика на черном (ниже порога).

Робот едет регулятором (ПД из урока 2) по часовой стрелке по дуге R 700 к центральному перекрестку X — там
пересекаются две кривые линии — и проезжает его. График — L и R с порогом 300: на перекрестке оба ниже порога.
Камера повернута: на старте робот едет вверх.
"""
from nano_base import NanoScene, render  # первым: добавляет в путь корень репозитория

import numpy as np
from manim import ValueTracker

import model
from model import fit_run
from scenes.base import MOTION, RunClock
from n3a_i_offset import PD
from nano_track import S_X_IN, ShortTrack
from params import Controller, Motor
from visuals.panel import GraphPanel, Signal

START = S_X_IN - 367.0  # дуга R 700, до перекрестка X 367 мм
INTRO = 0.8


class Nano3b(NanoScene):
    def runs(self, track):
        """Прогон и докатывание — MOTION = 8 с видео."""
        return fit_run(lambda t: model.cached_simulate(track, track.pose_at(START), ctrl=Controller(**PD), t_end=t,
                                                       sensor=self.sensor),
                       lambda run: model.brake(track, run, geom=self.geom, sensor=self.sensor, motor=Motor()),
                       MOTION * self.speed)

    def construct(self):
        track = ShortTrack()
        run, stop = self.runs(track)
        self.show(track, run)
        a0, a1 = model.channel_sides(self.sensor)
        trs = Controller().threshold
        t_max = run.t[-1] + stop.t[-1]
        panel = GraphPanel(self.fr, self.st, "Перекресток", "едем по линии", y=("АЦП", 0, 1023), t_max=t_max * 1.06,
                           t_ticks=np.arange(0, t_max, 0.5),
                           signals=[Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")],
                           formula=f"L < {trs:g} && R < {trs:g}", formula_colors=[("L", "L"), ("R", "R")])
        limit = panel.plot.hline(trs, f"порог {trs:g}", ValueTracker(1.0), side="left")
        clock = RunClock(self.t)
        self.add(panel, limit, panel.curves(run, clock))
        self.play(panel.intro(), run_time=INTRO)
        self.begin_segment()

        both = (run.L < trs) & (run.R < trs)
        t_in, t_out = float(run.tt[np.argmax(both)]), float(run.tt[len(both) - 1 - np.argmax(both[::-1])])
        self.advance(t_in)
        self.advance(t_in + 0.8 * self.speed, panel.set_caption("оба ниже порога — перекресток"))
        self.advance(max(t_out + 1.2 * self.speed, t_in + 2.0 * self.speed))
        self.advance(run.t[-1], panel.set_caption("едем дальше"))
        clock.stop()
        self.switch(stop)
        self.advance(stop.t[-1])
        self.end_segment()


if __name__ == "__main__":
    render(Nano3b, "480p")
