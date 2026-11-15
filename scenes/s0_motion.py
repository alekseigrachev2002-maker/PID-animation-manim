"""Сцена 0. Движение робота: команды моторам → как едет робот. Без датчиков и без линии.

Три маневра, каждый с места и до остановки (после команды — motors(0, 0) и докатывание по физике):
разгон с линейным ростом ШИМ до 255 и удержанием, разворот на месте ровно на 180°, левое колесо быстрее —
робота уводит вправо по дуге. Камера едет за роботом по полю 2400×1500 (старт — на вертикали «0–5» между желтым
квадратом и центральным перекрестком): линии — эталон «прямо», датчиков у робота нет. Между маневрами робота
переставляют «рукой». Проигрывание ×0.4.
На графике — только то, что пишется в коде: ШИМ левого и правого мотора.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import FadeOut  # noqa: E402

import model  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from track import field1  # noqa: E402
from visuals.panel import GraphPanel, Signal  # noqa: E402

T_RAMP = 0.35  # разгон: ШИМ линейно 0 → 255 за T_RAMP, с
T_HOLD = 0.6   # затем держим 255 (камера за роботом — место в кадре не ограничивает)

# пояснение в шапке, вызов в блоке формулы и цвета его частей, старт (x, y, ψ) в мм от CENTER, ШИМ, время команды
MANEUVERS = [
    dict(caption="разгон до 255", code="motors(pwm, pwm)", colors=[("pwm", "L", 0), ("pwm", "R", 1)],
         start=(0.0, -95.0, 0.0), pwm=lambda t: (255 * min(t / T_RAMP, 1.0),) * 2, t_on=T_RAMP + T_HOLD),
    dict(caption="разворот на месте", code="motors(-150, 150)", colors=[("-150", "L"), ("150", "R", 1)],
         start=(0.0, 0.0, 0.0), pwm=(-150, 150), t_on=None),          # None — подобрать: с докатыванием ровно 180°
    dict(caption="левое колесо быстрее", code="motors(220, 180)", colors=[("220", "L"), ("180", "R")],
         start=(-25.0, -95.0, 0.0), pwm=(220, 180), t_on=2.0),        # дуга вправо, курс около −30°
]

CENTER = (601.7, 600.0)       # точка поля 1, от которой отсчитаны старты: вертикаль x = 601.7, между квадратом и перекрестком


class Scene0(PIDScene):
    speed = 0.4           # проигрывание в 2,5 раза медленнее
    t_slide = 1.0         # перестановка робота между маневрами, с
    t_turn = 1.0

    def half_turn(self, track, pose, pwm):
        """Время команды разворота, после которого робот с докатыванием повернется ровно на 180° (по модели)."""
        lo, hi = 0.05, 5.0
        for _ in range(30):
            mid = (lo + hi) / 2
            psi = model.drive(track, pose, pwm, mid, stop=True, geom=self.geom).psi[-1] - pose[2]
            lo, hi = (mid, hi) if abs(psi) < np.pi else (lo, mid)
        return (lo + hi) / 2

    def construct(self):
        fr, st = self.fr, self.st
        track = field1()
        starts = [(CENTER[0] + m["start"][0], CENTER[1] + m["start"][1], m["start"][2]) for m in MANEUVERS]
        t_on = [m["t_on"] if m["t_on"] is not None else self.half_turn(track, p, m["pwm"])
                for p, m in zip(starts, MANEUVERS)]
        trajs = [model.drive(track, p, m["pwm"], t, stop=True, geom=self.geom)
                 for p, m, t in zip(starts, MANEUVERS, t_on)]
        self.show(track, trajs[0], sensors=False)       # камера за роботом, как в остальных сценах

        t_max = max(tr.t[-1] for tr in trajs) * 1.12
        panel = GraphPanel(fr, st, "Движение робота", MANEUVERS[0]["caption"], y=("ШИМ", -255, 255),
                           t_max=t_max, t_ticks=np.arange(0, t_max, 0.5),
                           signals=[Signal("левый мотор", "pwmL", "L"), Signal("правый мотор", "pwmR", "R")],
                           formula=MANEUVERS[0]["code"], formula_colors=MANEUVERS[0]["colors"])   # текст — сразу
        self.add(panel)
        self.play(panel.intro(), run_time=0.8)          # заголовок выезжает слева

        curves = None
        for i, (m, traj) in enumerate(zip(MANEUVERS, trajs)):
            if i > 0:
                self.play(FadeOut(curves), panel.set_text(m["code"], m["colors"], m["caption"]), run_time=0.8)
                self.return_to(starts[i], self.t_slide, self.t_turn)
                self.switch(traj)
            curves = panel.curves(traj, self.t)
            self.add(curves)
            self.wait(0.3)
            self.advance(traj.t[-1])
            self.wait(0.8)
        self.wait(1.0)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm. Качество: "240p", "480p", "720p", "1080p", "4k"; still=True — только последний кадр.
    # render(Scene0, "1080p", still=True)
    render(Scene0)
