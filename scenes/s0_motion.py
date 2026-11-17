"""Сцена 0. Движение робота: команды моторам → как едет робот. Без датчиков и без линии.

Три маневра, каждый с места и до остановки (после команды — motors(0, 0) и докатывание по физике):
разгон с линейным ростом ШИМ до 255 и удержанием, оборот на месте ровно на 360°, левое колесо быстрее —
робота уводит вправо по короткой дуге. Каждый маневр — один промежуток BLOCK. Камера неподвижна и смотрит на
фрагмент поля 2400×1500 (вертикаль «0–5» между желтым квадратом и центральным перекрестком): линия — эталон
«прямо», датчиков у робота нет. Между маневрами робота переставляют «рукой» (HAND). Езда — PLAYBACK.
На графике — только то, что пишется в коде: ШИМ левого и правого мотора.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from manim import FadeOut  # noqa: E402

import model  # noqa: E402
from scenes.base import PIDScene, render  # noqa: E402
from track import field1  # noqa: E402
from visuals.panel import GraphPanel, Signal  # noqa: E402

T_RAMP = 0.95  # разгон: ШИМ линейно 0 → 255 за T_RAMP, с (медленный рост — видно, как растет скорость)
T_HOLD = 0.0   # удержания 255 нет: дальше не пускает рамка 80 % правой панели (камера неподвижна)

# пояснение в шапке, вызов в блоке формулы и цвета его частей, старт (x, y, ψ) в мм от CENTER, ШИМ, время команды
MANEUVERS = [
    dict(caption="разгон до 255", code="motors(pwm, pwm)", colors=[("pwm", "L", 0), ("pwm", "R", 1)],
         start=(0.0, -100.0, 0.0), pwm=lambda t: (255 * min(t / T_RAMP, 1.0),) * 2, t_on=T_RAMP + T_HOLD),
    dict(caption="оборот на месте", code="motors(-200, 200)", colors=[("-200", "L"), ("200", "R", 1)],
         start=(0.0, 0.0, 0.0), pwm=(-200, 200), t_on=None, turn=360),  # None — подобрать: с докатыванием ровно turn°
    dict(caption="левое колесо быстрее", code="motors(100, 75)", colors=[("100", "L"), ("75", "R")],
         start=(-26.0, -80.0, 0.0), pwm=(100, 75), t_on=1.6),         # медленная дуга вправо: ~160 мм вперед, курс −36°
]

CENTER = (601.7, 600.0)       # точка поля 1 в центре правой панели: вертикаль x = 601.7, между квадратом и перекрестком
PANEL_PX = (1066.5, 360.0)    # центр правой панели — сюда неподвижная камера ставит CENTER
INSIDE = 0.8                  # робот целиком в этой доле правой панели (центр: ±27 мм по x, ±100 мм по y)


class Scene0(PIDScene):
    t_slide = 1.0         # перестановка робота между маневрами, с
    t_turn = 1.0

    def check_inside(self, traj):
        """Предупредить, если робот (круг шасси) выходит за INSIDE правой панели."""
        lay, mm = self.fr.layout, self.fr.layout.px_per_mm
        w, h = lay.width_px - lay.graph_w_px, self.fr.height_px
        half_w, half_h = INSIDE * w / 2 / mm - self.geom.chassis_r, INSIDE * h / 2 / mm - self.geom.chassis_r
        dx, dy = np.abs(traj.x - CENTER[0]).max(), np.abs(traj.y - CENTER[1]).max()
        if dx > half_w + 0.5 or dy > half_h + 0.5:
            warnings.warn(f"робот выходит за {INSIDE:.0%} панели: |x| ≤ {dx:.0f} (можно {half_w:.0f}), "
                          f"|y| ≤ {dy:.0f} (можно {half_h:.0f}) мм")
        return dx <= half_w + 0.5 and dy <= half_h + 0.5

    def turn_time(self, track, pose, pwm, turn_deg=180.0):
        """Время команды поворота на месте, после которого робот с докатыванием повернется ровно на turn_deg."""
        lo, hi = 0.05, 8.0
        for _ in range(30):
            mid = (lo + hi) / 2
            psi = model.drive(track, pose, pwm, mid, stop=True, geom=self.geom).psi[-1] - pose[2]
            lo, hi = (mid, hi) if abs(psi) < np.radians(turn_deg) else (lo, mid)
        return (lo + hi) / 2

    def construct(self):
        fr, st = self.fr, self.st
        track = field1()
        starts = [(CENTER[0] + m["start"][0], CENTER[1] + m["start"][1], m["start"][2]) for m in MANEUVERS]
        t_on = [m["t_on"] if m["t_on"] is not None else self.turn_time(track, p, m["pwm"], m.get("turn", 180))
                for p, m in zip(starts, MANEUVERS)]
        trajs = [model.drive(track, p, m["pwm"], t, stop=True, geom=self.geom)
                 for p, m, t in zip(starts, MANEUVERS, t_on)]
        for tr in trajs:
            self.check_inside(tr)
        self.show(track, trajs[0], camera="fixed", center=CENTER, anchor_px=PANEL_PX, sensors=False)   # камера стоит

        t_max = max(tr.t[-1] for tr in trajs) * 1.12
        panel = GraphPanel(fr, st, "Движение робота", MANEUVERS[0]["caption"], y=("ШИМ", -255, 255),
                           t_max=t_max, t_ticks=np.arange(0, t_max, 0.5),
                           signals=[Signal("левый мотор", "pwmL", "L"), Signal("правый мотор", "pwmR", "R")],
                           formula=MANEUVERS[0]["code"], formula_colors=MANEUVERS[0]["colors"])   # текст — сразу
        self.add(panel)
        self.play(panel.intro(), run_time=0.8)          # заголовок выезжает слева

        curves = None
        for i, (m, traj) in enumerate(zip(MANEUVERS, trajs)):     # каждый маневр — один промежуток BLOCK
            if i > 0:
                self.play(FadeOut(curves), panel.set_text(m["code"], m["colors"], m["caption"]), run_time=0.8)
                self.return_to(starts[i], self.t_slide, self.t_turn)
                self.switch(traj)
            curves = panel.curves(traj, self.t)
            self.add(curves)
            self.advance(traj.t[-1])
            self.end_block()


if __name__ == "__main__":
    # Кнопка ▶ в PyCharm: видео в videos/Scene0_<качество>.mp4. Качество: "240p", "480p", "720p", "1080p", "4k".
    render(Scene0, "480p")
    # render(Scene0, "1080p")                  # для презентации; все шесть сразу — scenes/lessons.py
