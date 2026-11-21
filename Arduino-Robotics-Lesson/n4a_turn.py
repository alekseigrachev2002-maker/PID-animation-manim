"""Урок 4, сцена 4a. Поворот на перекрестке к кубику: датчики линии + дальномер HC-SR04 (новый элемент робота).

Задача урока: робот едет по линии от старта по часовой стрелке, доезжает до главного перекрестка X, тормозит,
поворачивает налево или направо — в зависимости от того, где кубик 5 см, — и едет до финиша. Направо — через
S-изгиб и левую петлю к финишу на внутренней прямой (2,1 м), налево — по дуге R 700 и внешней петле к финишу
на внешней прямой (2,7 м).

Сцена (кубик на правой ветке, 16 см от X до центра — см. storyboard):
1. регулятор ведет робота по S-изгибу к X; оба датчика ниже порога — motors(0, 0), робот докатывается;
2. робот доезжает осью колес на X и поворачивает на месте направо, на ветку; дальномер видит кубик — ~5 см;
3. dist < 10 — кубик справа: разворот на месте налево (180°) и дальше по левой ветке регулятором.
Подсцены 1 и 3 — 8 с движения (MOTION), подсцена 2 — сколько займут доезд и поворот. График — dist, см
(без кубика в луче — верх шкалы), порог 10 см. Камера за роботом, повернута: на старте робот едет вверх;
так на X правая ветка с кубиком смотрит вверх, левая — вниз, и все помещается в обычном масштабе.
"""
from nano_base import MOTION, NanoScene, RunClock, render  # первым: добавляет в путь корень репозитория

import numpy as np
from manim import ValueTracker

import model
import nano_model
from n3a_i_offset import PD
from nano_robot import Cube, SonarRobot, sonar_distance
from nano_track import ShortTrack
from params import Controller, Layout, Motor
from visuals.layout import Frame
from visuals.panel import GraphPanel, Signal

S_X = 4847.0          # перекресток X по пути от старта (по часовой)
S_OTHER = 617.0       # та же точка X на другой линии (по ней — ветки направо и налево)
CUBE_ALONG = 160.0    # кубик на линии правой ветки, от X до центра: разворот на X с зазором ~3 см (см. storyboard)
CUBE_SIDE = 50.0
CUBE_COLOR = ("#22C55E", "#166534")   # заливка и контур: зеленый (выбран из 6 вариантов)
LIMIT_CM = 10         # порог дальномера: на X кубик в ~5 см, на подъезде по S-изгибу дальномер видит его с 18 см
MAX_CM = 60           # верх шкалы графика: «кубика нет»
PWM_TURN = 150        # поворот на месте: motors(150, -150) — направо, motors(-150, 150) — налево
PWM_CREEP = 120       # доезд осью колес на X
DIST = Signal("dist — дальномер, см", "dist", "#D7DCE5")


def cube_pose(track):
    """Центр и поворот кубика: центр — на линии правой ветки; грань, на которую смотрит датчик, перпендикулярна
    черной линии в центре кубика (направление линии — по точкам ±10 мм от центра)."""
    s = S_OTHER + CUBE_ALONG
    (x0, y0, _), (x1, y1, _) = track.route.pose_at(s - 10), track.route.pose_at(s + 10)
    x, y, _ = track.route.pose_at(s)
    return np.array([x, y]), float(np.arctan2(-(x1 - x0), y1 - y0))


def end_pose(tr):
    return float(tr.x[-1]), float(tr.y[-1]), float(tr.psi[-1])


class Nano4a(NanoScene):
    robot_cls = SonarRobot

    # ---------- траектория ----------

    def bisect(self, f, lo, hi, n=30):
        """Наименьшее t в [lo, hi], при котором f(t) истинно (f монотонна)."""
        for _ in range(n):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if f(mid) else (mid, hi)
        return (lo + hi) / 2

    def drive(self, pose, pwm, t):
        return model.drive(self.track, pose, pwm, t, stop=True, geom=self.geom, sensor=self.sensor)

    def approach(self):
        """Подсцена 1: регулятор от точки на S-изгибе до X (стоп по порогу) и докатывание — MOTION секунд видео.
        Подбирается, откуда стартовать."""
        thr = Controller().threshold
        target = MOTION * self.speed

        def total(back):
            run = nano_model.cached_simulate(self.track, self.track.pose_at(S_X - back), ctrl=Controller(**PD),
                                             t_end=6.0, stop_below=thr)
            stop = model.brake(self.track, run, geom=self.geom, sensor=self.sensor, motor=Motor())
            return run, stop

        back = self.bisect(lambda b: sum(x.t[-1] for x in total(round(b))) >= target, 150.0, 1500.0, n=16)
        return total(round(back))

    def turn_to(self, pose, psi_target, pwm):
        """Поворот на месте с выбегом до курса psi_target (рад)."""
        def reached(t):
            d = self.drive(pose, pwm, t).psi[-1] - pose[2]
            want = (psi_target - pose[2] + np.pi) % (2 * np.pi) - np.pi
            want = want if np.sign(want) == np.sign(pwm[1] - pwm[0]) else want + np.sign(pwm[1] - pwm[0]) * 2 * np.pi
            return abs(d) >= abs(want)
        return self.drive(pose, pwm, self.bisect(reached, 0.02, 4.0))

    def creep(self, pose, x_target):
        """Доезд прямо, пока ось колес (с выбегом) не встанет на точку x_target."""
        fwd = np.array([-np.sin(pose[2]), np.cos(pose[2])])
        need = float((np.asarray(x_target) - np.array(pose[:2])) @ fwd)

        def passed(t):
            tr = self.drive(pose, (PWM_CREEP, PWM_CREEP), t)
            return float((np.array([tr.x[-1], tr.y[-1]]) - np.array(pose[:2])) @ fwd) >= need
        return self.drive(pose, (PWM_CREEP, PWM_CREEP), self.bisect(passed, 0.02, 3.0))

    def leave(self, pose):
        """Подсцена 3: разворот налево на 180° и регулятор по левой ветке, с докатыванием — MOTION секунд видео."""
        left = self.track.route.pose_at(S_OTHER)[2] + np.pi
        spin = self.turn_to(pose, left, (-PWM_TURN, PWM_TURN))
        rest = MOTION * self.speed - spin.t[-1]
        start = end_pose(spin)

        def total(t):
            run = nano_model.cached_simulate(self.track, start, ctrl=Controller(**PD), t_end=round(t, 3))
            return run, model.brake(self.track, run, geom=self.geom, sensor=self.sensor, motor=Motor())

        t = self.bisect(lambda t: sum(x.t[-1] for x in total(t)) >= rest, 0.2, rest)
        return [spin, *total(t)]

    def plan(self):
        x_pt = np.array(self.track.route.pose_at(S_X)[:2])
        run, stop = self.approach()
        creep = self.creep(end_pose(stop), x_pt)
        right = self.turn_to(end_pose(creep), self.track.route.pose_at(S_OTHER)[2], (PWM_TURN, -PWM_TURN))
        out = self.leave(end_pose(right))
        return [[run, stop], [creep, right], out]

    # ---------- сцена ----------

    def setup(self):
        super().setup()
        self.track = ShortTrack()
        center, angle = cube_pose(self.track)
        self.cube_spec = (center, angle)

    def field_items(self, view):
        center, angle = self.cube_spec
        self.cube = Cube(view, center, CUBE_SIDE, angle, *CUBE_COLOR)
        return [self.cube]

    def construct(self):
        segs = self.plan()
        parts = [p for seg in segs for p in seg]
        traj = model.chain(*parts)
        cubes = [(self.cube_spec[0], CUBE_SIDE)]
        poses = np.array([[traj.at(t)[k] for k in ("x", "y", "psi")] for t in traj.tt])
        traj.dist = np.array([min(sonar_distance(p, cubes), MAX_CM * 10) for p in poses]) / 10    # см
        ends = np.cumsum([sum(p.t[-1] for p in seg) for seg in segs])
        self.show(self.track, traj)

        trs = LIMIT_CM
        panel = GraphPanel(self.fr, self.st, "Кубик на перекрестке", "едем до перекрестка", y=("dist, см", 0, MAX_CM),
                           t_max=traj.t[-1] * 1.04, t_ticks=np.arange(0, traj.t[-1], 1.0), signals=[DIST],
                           formula=f"if (dist < {trs}) налево; else направо;", formula_colors=[("dist", DIST.color)])
        limit = panel.plot.hline(trs, f"{trs} см", ValueTracker(1.0), side="left")
        clock = RunClock(self.t)
        self.add(panel, limit, panel.curves(traj, clock))
        self.play(panel.intro(), run_time=0.8)

        # 1: по линии до X, стоп по порогу датчиков линии
        self.begin_segment()
        self.advance(ends[0])
        self.end_segment()
        # 2: доезд на X и поворот направо — dist
        d_right = float(traj.dist[np.searchsorted(traj.tt, ends[1]) - 1])
        self.play(panel.set_caption("поворот направо"), run_time=0.8)
        self.begin_segment()
        self.advance(ends[1])
        self.end_segment(check=False)
        self.play(panel.set_caption(f"dist = {d_right:.0f} см: кубик"), run_time=0.8)
        # 3: налево, по левой ветке
        self.play(panel.set_caption("налево, к финишу"), run_time=0.8)
        self.begin_segment()
        self.advance(ends[2])
        clock.stop()
        self.end_segment()


class Nano4aCloseUp(Nano4a):
    """Картинка: крупный план крепления дальномера на шасси и кубик (масштаб ×2,5 к сценам). Только последний кадр."""

    def setup(self):
        super().setup()
        self.fr = Frame(Layout(px_per_mm=4.0))

    def construct(self):
        x, y, _ = self.track.route.pose_at(S_X)
        psi = self.track.route.pose_at(S_OTHER)[2]
        traj = model.prescribed(self.track, np.array([0.0, 0.01]), x, y, psi, sensor=self.sensor)
        center = np.array([x, y]) + np.array([-np.sin(psi), np.cos(psi)]) * 70
        self.show(self.track, traj, camera="fixed", center=center, anchor_px=(1066.5, 360.0), rot=-psi)
        self.wait(0.1, frozen_frame=False)


if __name__ == "__main__":
    render(Nano4a, "480p")
    # render(Nano4aCloseUp, "720p", still=True)     # крупный план крепления
