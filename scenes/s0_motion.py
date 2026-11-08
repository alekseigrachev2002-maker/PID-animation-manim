"""Сцена 0. Движение робота: команды моторам → как едет робот. Без датчиков и без линии.

Три манёвра, каждый с места и до остановки (после команды — motors(0, 0) и докатывание по физике):
разгон с линейным ростом ШИМ до 255 и удержанием, разворот на месте, левое колесо чуть быстрее — робота уводит вправо.
Камера неподвижна; между манёврами робота переставляют «рукой». На графике — только то, что пишется в коде:
ШИМ левого и правого мотора.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import FadeOut  # noqa: E402

import model  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from track import Track  # noqa: E402
from visuals.panel import GraphPanel, Signal  # noqa: E402

T_RAMP = 0.35  # разгон: ШИМ линейно 0 → 255 за T_RAMP, с
T_HOLD = 0.15  # затем держим 255 — дольше не даёт место в кадре (на 255 робот едет ~615 мм/с)

# пояснение в шапке, вызов в блоке формулы и цвета его частей, старт (x, y, ψ) в мм от центра панели, ШИМ, время команды
MANEUVERS = [
    dict(caption="разгон до 255", code="motors(pwm, pwm)", colors=[("pwm", "L", 0), ("pwm", "R", 1)],
         start=(0.0, -95.0, 0.0), pwm=lambda t: (255 * min(t / T_RAMP, 1.0),) * 2, t_on=T_RAMP + T_HOLD),
    dict(caption="разворот на месте", code="motors(-150, 150)", colors=[("-150", "L"), ("150", "R", 1)],
         start=(0.0, 0.0, 0.0), pwm=(-150, 150), t_on=0.65),          # с докатыванием — ровно 180°
    dict(caption="левое колесо быстрее", code="motors(180, 150)", colors=[("180", "L"), ("150", "R")],
         start=(-25.0, -95.0, 0.0), pwm=(180, 150), t_on=0.55),       # уводит вправо на ~37 мм, курс −23°
]

PANEL_PX = (1066.5, 360.0)    # центр правой панели — сюда камера ставит точку поля (0, 0)
INSIDE = 0.8                  # робот целиком в этой доле правой панели


class Scene0(PIDScene):
    speed = 0.2           # манёвры короткие (место в кадре) — проигрываем впятеро медленнее
    t_slide = 1.0         # перестановка робота между манёврами, с
    t_turn = 1.0

    def check_inside(self, traj):
        """Предупредить, если робот (круг шасси) выходит за INSIDE правой панели."""
        lay, mm = self.fr.layout, self.fr.layout.px_per_mm
        w, h = lay.width_px - lay.graph_w_px, self.fr.height_px
        half_w, half_h = INSIDE * w / 2 / mm - self.geom.chassis_r, INSIDE * h / 2 / mm - self.geom.chassis_r
        if np.abs(traj.x).max() > half_w or np.abs(traj.y).max() > half_h:
            warnings.warn(f"робот выходит за {INSIDE:.0%} панели: |x| ≤ {np.abs(traj.x).max():.0f} (можно {half_w:.0f}), "
                          f"|y| ≤ {np.abs(traj.y).max():.0f} (можно {half_h:.0f}) мм")

    def construct(self):
        fr, st = self.fr, self.st
        track = Track()                                    # линии нет
        trajs = [model.drive(track, m["start"], m["pwm"], m["t_on"], stop=True, geom=self.geom) for m in MANEUVERS]
        for tr in trajs:
            self.check_inside(tr)
        self.show(track, trajs[0], camera="fixed", anchor_px=PANEL_PX, sensors=False)

        t_max = max(tr.t[-1] for tr in trajs) * 1.12
        panel = GraphPanel(fr, st, "Движение робота", MANEUVERS[0]["caption"], y=("ШИМ", -255, 255),
                           t_max=t_max, t_ticks=[0, 0.5, 1],
                           # при равных командах правая (оранжевая) лежит поверх левой — в шапке и так motors(pwm, pwm)
                           signals=[Signal("левый мотор", "pwmL", "L"), Signal("правый мотор", "pwmR", "R")])
        self.add(panel)
        self.play(panel.set_formula(MANEUVERS[0]["code"], MANEUVERS[0]["colors"]), run_time=0.8)

        curves = None
        for i, (m, traj) in enumerate(zip(MANEUVERS, trajs)):
            if i > 0:
                self.play(FadeOut(curves), panel.set_caption(m["caption"]),
                          panel.set_formula(m["code"], m["colors"]), run_time=0.8)
                self.return_to(m["start"], self.t_slide, self.t_turn)
                self.switch(traj)
            curves = panel.curves(traj, self.t)
            self.add(curves)
            self.wait(0.3)
            self.advance(traj.t[-1])
            self.wait(0.8)
        self.wait(1.0)


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: черновое видео 480p и сразу открыть.
    # Кадр png 1080p: render(Scene0, "high_quality", still=True)
    render(Scene0)
