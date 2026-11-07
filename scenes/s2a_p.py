"""Сцена 2a. Подбор P на прямой: нет регуляции → едет по линии → перерегулирование.

Прогоны идут подряд по одной линии. Между ними — возврат: стоп, если робот съехал, линия подъезжает
обратно, затем робот поворачивается на стартовый угол.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import FadeOut  # noqa: E402

import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Motor  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from visuals.panels import Header, MotorReadout, Plot, legend  # noqa: E402

U_P = "control = kp·err"

# (kp, пояснение в шапке) — в порядке показа
RUNS = [
    (0.0, "kp = 0, нет регуляции"),
    (0.3, "kp = 0.3, едет по линии"),
    (0.8, "kp = 0.8, перерегулирование"),
]


class Scene2a(PIDScene):
    runs = RUNS         # прогоны (kp, пояснение)
    psi0_deg = 15.0     # стартовый поворот относительно линии
    delay = 0.020       # задержка команда → колесо, с (заглушка до замеров)
    t_run = 2.0         # длительность прогона, с
    speed = 0.33        # проигрывание прогонов втрое медленнее
    t_slide = 1.0       # линия подъезжает под робота, с
    t_turn = 1.0        # поворот на стартовый угол, с

    def run(self, line, pose, kp):
        return model.cached_simulate(line, tuple(float(v) for v in pose), ctrl=Controller(kp=kp),
                                     motor=Motor(delay=self.delay), t_end=self.t_run)

    def reset(self, line):
        """Возврат перед следующим прогоном: всегда точно на центр линии, затем поворот на стартовый угол.

        На центр — всегда: иначе поворот на 15° при смещении в мёртвой зоне может увести оба датчика
        по одну сторону линии, и прогон начнётся с потерянной линией.
        """
        _, y, _ = self.end_pose()
        return self.return_to(line, (0.0, y, np.radians(self.psi0_deg)), self.t_slide, self.t_turn)

    def construct(self):
        fr, st = self.fr, self.st
        line = track.straight(y1=20000)
        pose = (0.0, 0.0, np.radians(self.psi0_deg))
        traj = self.run(line, pose, self.runs[0][0])
        leg = legend(fr, [(st.e, "err = L − R", 1)], st)
        readout = MotorReadout(fr, leg.get_right()[0] / fr.unit + fr.layout.width_px / 2 + 40,
                               model.wheel_signs(Motor()), st=st)
        self.bind(traj, line, readout=readout)

        plot = Plot(fr, t_max=self.t_run * 1.12, lo=-1023, hi=1023, st=st)   # ось чуть длиннее прогона
        ticks = {v: (lambda lo, hi: 1) for v in (-1023, 0, 1023)}
        header = Header(fr, "Подбираем P", self.runs[0][1], st)
        self.add(plot.axes(ticks, y_label="err", t_ticks=[0, 0.5, 1, 1.5, 2]), header, leg)
        self.play(header.set_formula(U_P, st.e), run_time=0.8)

        for i, (kp, caption) in enumerate(self.runs):
            if i > 0:
                pose = self.reset(line)
                traj = self.run(line, pose, kp)
                self.switch(traj)
                self.play(readout.opacity.animate.set_value(1.0), run_time=0.4)
            curve = plot.curve(traj.tt, traj.e, st.e, self.t, width_px=3)
            self.add(curve)
            self.wait(0.3)
            self.advance(traj.t[-1])
            self.wait(0.8)
            if i < len(self.runs) - 1:
                # стоп: кривая гаснет, числа ШИМ приглушаются, в шапке — следующий прогон
                self.play(FadeOut(curve), readout.opacity.animate.set_value(0.35),
                          header.set_caption(self.runs[i + 1][1]), run_time=0.8)
        self.wait(1.0)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p: render(Scene2a, "high_quality", still=True)
    render(Scene2a)
