"""Урок 1, сцена 1a. Движение: команды моторам → как едет робот. Без датчиков.

Три маневра, по одному на промежуток, друг за другом без перестановок: разгон с ростом ШИМ до 255 и остановка
(после motors(0, 0) робот катится по инерции); оборот на месте на 360°; разные скорости колес — дуга вправо.
Камера едет за роботом (повернута: на старте робот едет вверх) — по верхней внутренней прямой трека, линия —
эталон «прямо». Длительность маневра подобрана под промежуток: разгон — время роста ШИМ, дуга — время команды.
"""
from nano_base import BLOCK, SLACK, NanoScene, RunClock, render  # первым: добавляет в путь корень репозитория

import numpy as np
from manim import FadeOut

import model
from nano_track import ShortTrack
from visuals.panel import GraphPanel, Signal

PLACE = 2250.0      # путь по маршруту (по часовой), мм: начало верхней внутренней прямой (1,2 м прямой впереди)
T_TEXT = 0.8        # заголовок в начале / смена подписи между маневрами, с

# fit: «ramp» — подобрать время роста ШИМ, «on» — время команды, «turn» — время на turn° (с выбегом)
MANEUVERS = [
    dict(caption="разгон и остановка", code="motors(pwm, pwm)", colors=[("pwm", "L", 0), ("pwm", "R", 1)],
         fit="ramp"),
    dict(caption="оборот на месте", code="motors(-250, 250)", colors=[("-250", "L"), ("250", "R", 1)],
         pwm=(-250, 250), fit="turn", turn=360),
    dict(caption="правое колесо быстрее — влево", code="motors(150, 180)", colors=[("150", "L"), ("180", "R")],
         pwm=(150, 180), fit="on"),
]


class Nano1a(NanoScene):
    def drive(self, track, pose, m, t):
        if m["fit"] == "ramp":
            return model.drive(track, pose, lambda u: (255 * min(u / t, 1.0),) * 2, t, stop=True,
                               geom=self.geom, sensor=self.sensor)
        return model.drive(track, pose, m["pwm"], t, stop=True, geom=self.geom, sensor=self.sensor)

    def fit(self, track, pose, m):
        """Маневр на весь промежуток (минус смена текста и SLACK): подбор времени делением пополам."""
        target = (BLOCK - T_TEXT - SLACK) * self.speed
        lo, hi = 0.05, 6.0
        for _ in range(30):
            mid = (lo + hi) / 2
            tr = self.drive(track, pose, m, mid)
            if m["fit"] == "turn":
                done = abs(tr.psi[-1] - pose[2]) >= np.radians(m["turn"])
            else:
                done = tr.t[-1] >= target
            lo, hi = (lo, mid) if done else (mid, hi)
        return self.drive(track, pose, m, (lo + hi) / 2)

    def construct(self):
        fr, st = self.fr, self.st
        track = ShortTrack()
        trajs, pose = [], track.pose_at(PLACE)
        for m in MANEUVERS:
            trajs.append(self.fit(track, pose, m))
            pose = (trajs[-1].x[-1], trajs[-1].y[-1], trajs[-1].psi[-1])    # следующий — с места остановки
        self.show(track, trajs[0], sensors=False)

        t_max = max(tr.t[-1] for tr in trajs) * 1.08
        panel = GraphPanel(fr, st, "Движение робота", MANEUVERS[0]["caption"], y=("ШИМ", -255, 255),
                           t_max=t_max, t_ticks=np.arange(0, t_max, 0.5),
                           signals=[Signal("левый мотор", "pwmL", "L"), Signal("правый мотор", "pwmR", "R")],
                           formula=MANEUVERS[0]["code"], formula_colors=MANEUVERS[0]["colors"])
        self.add(panel)
        self.play(panel.intro(), run_time=T_TEXT)

        curves = None
        for i, (m, traj) in enumerate(zip(MANEUVERS, trajs)):
            if i > 0:
                self.play(FadeOut(curves), panel.set_text(m["code"], m["colors"], m["caption"]), run_time=T_TEXT)
                self.switch(traj)
            clock = RunClock(self.t)
            curves = panel.curves(traj, clock)
            self.add(curves)
            self.advance(traj.t[-1])
            clock.stop()
            self.end_block()


if __name__ == "__main__":
    render(Nano1a, "480p")
