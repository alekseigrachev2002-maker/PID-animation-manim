"""Основа сцен.

PIDScene — стенд: трекер времени t, камера, поле, робот и элементы, которым каждый кадр раздается состояние.
    Атрибуты сцены: sensor (датчики), robot_cls (робот, например с дальномером), rot (поворот камеры на всю сцену:
    число или "start" — стартовый курс робота вверх), fast (ускорение простой езды), panel_bg (темная панель).
    Время: промежутки BLOCK (end_block — сцена ровно N·BLOCK) или подсцены (begin_segment / end_segment — моменты
    концов подсцен в SEGMENTS, для картинок уроков). advance_fast — проигрывание с ускорением вне зон, hand_to —
    перестановка «рукой» с одной скоростью для всех сцен.
RunSeries — серия прогонов регулятора: прогон → стоп → возврат → следующий (олимпиадные сцены 2a–2c).
MotionRunSeries — то же, но прогон с докатыванием — MOTION секунд видео, возврат «рукой» сверх (новый курс).
render() — запуск кнопкой ▶ в PyCharm (качество, видео или последний кадр).
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from manim import FadeOut, Scene, ValueTracker, config, linear, tempconfig  # noqa: E402

import model  # noqa: E402
from params import Controller, Geometry, Motor, Sensor, Style  # noqa: E402
from visuals.camera import Camera  # noqa: E402
from visuals.field import Field  # noqa: E402
from visuals.layout import Frame, backdrop  # noqa: E402
from visuals.panel import ERR, GraphPanel  # noqa: E402
from visuals.robot import Robot  # noqa: E402

GENERAL = 0.75  # общий коэффициент замедления (доля реального времени): крутить его, чтобы ускорить или замедлить все
PLAYBACK = 0.4 * GENERAL  # езда во всех сценах (сейчас 0.3 — в 3,3 раза медленнее реального времени)
BLOCK = 11.0  # длина одного промежутка сцены, с: сцена — целое число промежутков (0 — 3, 1 — 2, 2a — 3, 2b/2c — 2, 3 — 1)
HAND = 0.85 * GENERAL  # движения «рукой»: перестановка, линия подъезжает, поворот на месте, качание в сцене 1
MOTION = 8.0  # подсцена: движение, с видео (прогон с докатыванием, маневр); заголовок, подписи, «рука» — сверх
SEGMENTS = {}  # имя сцены → моменты концов подсцен в видео, с (заполняется при рендере; картинки уроков)
FAST_TAU = 0.5  # плавность перехода ускорения: постоянная EMA, с времени модели
HAND_V = 150.0  # «рука»: средняя скорость переноса, мм/с (время модели; на экране — с замедлением HAND)
HAND_W = 120.0  # «рука»: средняя скорость поворота, °/с
HAND_MIN = 0.4  # самая короткая перестановка, с


def hand_time(a, b):
    """Время перестановки «рукой» из позы a в позу b (сдвиг и поворот вместе), с — одна скорость для всех сцен."""
    dpsi = abs((b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi)
    return max(np.hypot(b[0] - a[0], b[1] - a[1]) / HAND_V, np.degrees(dpsi) / HAND_W, HAND_MIN)


def warp(t0, t1, zones, fast, tau=FAST_TAU, speed=PLAYBACK, dt=0.002):
    """Ускорение проигрывания на участке модели [t0, t1]: в зонах zones = [(a, b), ...] (время модели) кратность 1,
    вне их — fast. Переход плавный: EMA вперед и назад по времени модели, берется меньшая — так к началу зоны
    кратность уже 1 и после зоны растет постепенно. Возвращает (длительность видео, rate_func для t, τ, k(τ))."""
    tau_grid = np.arange(t0, t1 + dt / 2, dt)
    target = np.full(len(tau_grid), float(fast))
    for a, b in zones:
        target[(tau_grid >= a) & (tau_grid <= b)] = 1.0
    alpha = 1 - np.exp(-dt / tau)
    fwd, bwd = target.copy(), target.copy()
    for i in range(1, len(target)):
        fwd[i] = fwd[i - 1] + (target[i] - fwd[i - 1]) * alpha
    for i in range(len(target) - 2, -1, -1):
        bwd[i] = bwd[i + 1] + (target[i] - bwd[i + 1]) * alpha
    k = np.maximum(np.minimum(fwd, bwd), 1.0)
    video = np.concatenate([[0.0], np.cumsum(dt / (speed * (k[1:] + k[:-1]) / 2))])
    total = float(video[-1])

    def rate(alpha_):
        return float((np.interp(alpha_ * total, video, tau_grid) - t0) / max(t1 - t0, 1e-9))
    return total, rate, tau_grid, k


def path_time(traj, dist):
    """Время модели, когда середина оси прошла dist мм от начала траектории (по пути колес)."""
    s = (traj.sL + traj.sR) / 2 - (traj.sL[0] + traj.sR[0]) / 2
    return float(np.interp(dist, s, traj.t))


class PIDScene(Scene):
    speed = PLAYBACK  # скорость проигрывания езды — одна для всех сцен; движения «рукой» — HAND
    sensor = Sensor()  # датчики линии сцены
    robot_cls = Robot  # робот (visuals.robot.SonarRobot — с дальномером)
    rot = 0.0  # поворот камеры, рад; "start" — стартовый курс робота смотрит вверх
    fast = 1.0  # ускорение простой езды (advance_fast); 1,0 — без ускорения
    panel_bg = True  # темная панель графиков слева; False — поле на весь кадр

    def play(self, *args, **kwargs):
        """Как Scene.play, плюс счет кадров видео (frames) — по тому же правилу, что у Manim: шагов 1/fps в run_time.
        Ждем всегда «на ходу» (frozen_frame=False), иначе Manim округляет паузу вниз и теряет кадр."""
        super().play(*args, **kwargs)
        self.frames = getattr(self, "frames", 0) + len(np.arange(0, self.duration, 1 / config.frame_rate))

    def end_block(self):
        """Конец промежутка: пауза до границы следующего BLOCK — ровно в кадрах (робот стоит, все видно).
        Длительность сцены — точно N·BLOCK в любом качестве. Если промежуток длиннее BLOCK — предупреждение."""
        self._blocks = getattr(self, "_blocks", 0) + 1
        need = round(self._blocks * BLOCK * config.frame_rate) - getattr(self, "frames", 0)
        if need < 0:
            warnings.warn(f"{type(self).__name__}: промежуток {self._blocks} длиннее BLOCK = {BLOCK} с "
                          f"на {-need / config.frame_rate:.1f} с")
        elif need > 0:
            self.wait((need - 0.5) / config.frame_rate, frozen_frame=False)   # ровно need кадров

    def setup(self):
        self.fr, self.st, self.geom = Frame(), Style(), Geometry()
        self.t = ValueTracker(0.0)
        self.add(backdrop(self.fr, self.st, self.panel_bg))

    # ---------- подсцены ----------

    def begin_segment(self):
        """Начало движения подсцены (после заголовка, смены подписи, перестановки «рукой»)."""
        self._seg_start = getattr(self, "frames", 0) / config.frame_rate

    def end_segment(self, check=True):
        """Конец подсцены: момент — в SEGMENTS (картинки урока берут кадр отсюда); если движение не MOTION —
        предупреждение (check=False — длина задана самим движением: качание «рукой», оборот на угол, ускорение)."""
        now = getattr(self, "frames", 0) / config.frame_rate
        SEGMENTS.setdefault(type(self).__name__, []).append(now)
        took = now - getattr(self, "_seg_start", now)
        if check and abs(took - MOTION) > 0.3:
            warnings.warn(f"{type(self).__name__}: движение в подсцене {len(SEGMENTS[type(self).__name__])} — "
                          f"{took:.1f} с (MOTION = {MOTION:g} с)")

    # ---------- стенд ----------

    def show(self, track, traj, camera="follow", center=(0.0, 0.0), anchor_px=None, sensors=True, items=(),
             rot=None):
        """Поставить на стенд поле, робота и элементы items (у каждого — update_state(state)).
        rot — поворот камеры (по умолчанию атрибут сцены rot; "start" — стартовый курс робота смотрит вверх).

        camera: "follow" или "fixed"; center — точка поля (мм) в точке экрана anchor_px (по умолчанию Layout.robot_px).
        """
        self.track, self.traj = track, traj
        rot = self.rot if rot is None else rot
        rot = -float(traj.psi[0]) if rot == "start" else rot
        self.view = Camera(self.fr, camera, center, None if anchor_px is None else self.fr.px(*anchor_px), rot)
        self.field = Field(self.view, track, self.sensor, self.st)
        self.robot = self.robot_cls(self.fr, self.geom, self.st, self.sensor, sensors)
        self.items = [m for m in items if m is not None]
        self.s_off = np.zeros(2)  # перенос пути колес между отрезками, чтобы протектор не прыгал

        def follow(_):
            s = self.state()
            self.view.follow(s["x"], s["y"])
            self.field.update_view()
            self.robot.place(self.view.point(s["x"], s["y"]), s["psi"] + self.view.rot)
            self.robot.set_wheel_phase(s["sL"], s["sR"]).set_signals(*s["levels"])
            for m in self.items:
                m.update_state(s)

        self.robot.add_updater(follow)
        follow(self.robot)
        for m in (self.field, *self.items):
            # их двигает апдейтер робота; пустой апдейтер говорит Manim перерисовывать их каждый кадр
            m.add_updater(lambda m: None)
        self.add(self.field, self.robot, *self.items)

    def state(self):
        """Состояние в момент t: поза, путь колес, сигналы регулятора, яркость меток датчиков."""
        t = self.t.get_value()
        s = {**self.traj.at(t), **self.traj.signals_at(t)}
        s["sL"], s["sR"] = np.array([s["sL"], s["sR"]]) + self.s_off
        self.wheel_s = np.array([s["sL"], s["sR"]])
        phys = np.array(model.to_channels(np.array([s["L"], s["R"]]), self.sensor))  # A0/A1 → левый/правый
        w, b = np.asarray(self.sensor.white, float), np.asarray(self.sensor.black, float)
        s["levels"] = (phys - w) / (b - w)
        return s

    def switch(self, traj, t=0.0):
        """Подменить проигрываемую траекторию (новый прогон или переход); путь колес продолжается.

        Проверка стыка: скорость и вращение робота до и после должны совпадать — иначе предупреждение.
        """
        (v0, w0), (v1, w1) = self.traj.speed_at(self.t.get_value()), traj.speed_at(t)
        # пороги с запасом над погрешностью оценки по позе (при резком разгоне — несколько мм/с);
        # настоящий скачок — это 0 → 300 мм/с или ~1 рад/с
        if abs(v1 - v0) > 30 or abs(w1 - w0) > 0.3:
            warnings.warn(f"скачок на стыке: скорость {v0:.0f} → {v1:.0f} мм/с, вращение {w0:.2f} → {w1:.2f} рад/с")
        p = traj.at(t)
        self.s_off = self.wheel_s - np.array([p["sL"], p["sR"]])
        self.traj = traj
        self.t.set_value(t)
        self.robot.update(0)

    def end_pose(self):
        p = self.traj.at(self.t.get_value())
        return p["x"], p["y"], p["psi"]

    def advance(self, t_to, *extra, speed=None):
        """Проиграть траекторию до момента t_to (плюс одновременные анимации extra)."""
        dur = (t_to - self.t.get_value()) / (speed or self.speed)
        self.play(self.t.animate.set_value(t_to), *extra, run_time=dur, rate_func=linear)

    def advance_fast(self, t_to, zones, *extra, fast=None):
        """Проиграть траекторию до t_to с ускорением fast (по умолчанию атрибут сцены) вне зон zones (см. warp);
        пометка self.rate (visuals.panel.RateLabel), если есть, показывает текущую кратность."""
        t0 = self.t.get_value()
        total, rate, tau, k = warp(t0, t_to, zones, self.fast if fast is None else fast, speed=self.speed)
        label = getattr(self, "rate", None)
        if label is not None:
            label.add_updater(lambda m: m.set_value(float(np.interp(self.t.get_value(), tau, k))))
        self.play(self.t.animate.set_value(t_to), *extra, run_time=total, rate_func=rate)
        if label is not None:
            label.clear_updaters()
            label.set_value(1.0)

    def hand_to(self, target, *extra):
        """Переставить робота «рукой» в позу target (сдвиг и поворот вместе, колеса не крутятся) — с одной
        скоростью для всех сцен (hand_time); extra — анимации на это же время (смена подписи)."""
        a = self.end_pose()
        dpsi = (target[2] - a[2] + np.pi) % (2 * np.pi) - np.pi
        t = hand_time(a, target)
        self.switch(model.move(self.track, a, (target[0], target[1], a[2] + dpsi), t, roll=False,
                               geom=self.geom, sensor=self.sensor))
        self.advance(t, *extra, speed=HAND)

    def return_to(self, target, t_slide=1.0, t_turn=1.0, together=False):
        """Возврат между прогонами: линия подъезжает так, что середина оси встает в target[:2] (колеса не крутятся),
        затем робот поворачивается на месте на курс target[2].

        together=True — одним движением: сдвиг и поворот одновременно, за t_slide + t_turn (по диагонали,
        а не «буквой Г»: для сцен на повороте, где курс в конце прогона далек от стартового)."""
        x, y, psi = self.end_pose()
        tx, ty, tpsi = target
        if together:
            dpsi = (tpsi - psi + np.pi) % (2 * np.pi) - np.pi
            self.switch(model.move(self.track, (x, y, psi), (tx, ty, psi + dpsi), t_slide + t_turn, roll=False,
                                   geom=self.geom, sensor=self.sensor))
            self.advance(t_slide + t_turn, speed=HAND)
            return
        if np.hypot(tx - x, ty - y) > 0.5:
            self.switch(model.move(self.track, (x, y, psi), (tx, ty, psi), t_slide, roll=False,
                                   geom=self.geom, sensor=self.sensor))
            self.advance(t_slide, speed=HAND)
        dpsi = (tpsi - psi + np.pi) % (2 * np.pi) - np.pi  # кратчайший поворот
        if abs(dpsi) > np.radians(0.5):
            self.switch(model.move(self.track, (tx, ty, psi), (tx, ty, psi + dpsi), t_turn,
                                   geom=self.geom, sensor=self.sensor))
            self.advance(t_turn, speed=HAND)


class RunSeries(PIDScene):
    """Серия прогонов регулятора по одной трассе. Сцена задает трассу, старт и прогоны:

        runs = [(0.15, "kp = 0.15, едет по линии"), (dict(kp=0.09, kd=1.0), "с D")]   # kp (только P) или dict
        def make_track(self): ...                 # трасса
        def start(self, i): ...                   # поза старта прогона i (для i > 0 — после возврата)
    """
    phrase = "Подбираем P"
    formula = "control = err * kp"  # как в скетче урока
    formula_colors = [("err", "e")]  # части формулы цветом своих кривых
    law = "control"  # в формулах колес: 150 ± control, как motors(speed + control, speed − control)
    runs = []
    signals = [ERR]
    y = ("err", -1023, 1023)
    t_run = 2.0  # длительность прогона, с
    t_ticks = [0, 0.5, 1, 1.5, 2]
    t_slide = 1.0  # возврат: линия подъезжает, с
    t_turn = 1.0  # возврат: поворот на месте, с
    return_together = False  # возврат одним движением (сдвиг + поворот) — для сцен на повороте

    def make_track(self):
        raise NotImplementedError

    def start(self, i):
        raise NotImplementedError

    def simulate(self, pose, ctrl):
        ctrl = Controller(**({"kp": ctrl, "kd": 0.0} if np.isscalar(ctrl) else ctrl))  # одно число — только P
        return model.cached_simulate(self.track, tuple(float(v) for v in pose), ctrl=ctrl,
                                     motor=Motor(), t_end=self.t_run)

    def construct(self):
        self.track = self.make_track()
        traj = self.simulate(self.start(0), self.runs[0][0])
        panel = GraphPanel(self.fr, self.st, self.phrase, self.runs[0][1], self.y, self.t_run * 1.12,
                           self.t_ticks, self.signals,
                           readout=dict(signs=model.wheel_signs(Motor()), law=self.law),
                           formula=self.formula, formula_colors=self.formula_colors)  # текст — сразу на месте
        self.show(self.track, traj, items=[panel.readout])
        self.add(panel)
        self.play(panel.intro(), run_time=0.8)  # заголовок выезжает слева
        readout = panel.readout
        curves = None
        for i, (ctrl, caption) in enumerate(self.runs):          # каждый прогон — один промежуток BLOCK
            if i > 0:
                self.play(FadeOut(curves), panel.set_caption(caption), run_time=0.8)   # кривые гаснут, новый прогон
                target = self.start(i)
                self.return_to(target, self.t_slide, self.t_turn, self.return_together)
                self.switch(self.simulate(target, ctrl))
                self.play(readout.opacity.animate.set_value(1.0), run_time=0.4)
            clock = RunClock(self.t)  # кривые растут за t во время прогона и замирают на его конце
            curves = panel.curves(self.traj, clock)
            self.add(curves)
            self.advance(self.traj.t[-1])
            clock.stop()
            # стоп: motors(0, 0), робот докатывается по той же физике; строка ШИМ приглушается — регулятор выключен
            self.switch(model.brake(self.track, self.traj, geom=self.geom, sensor=self.sensor,
                                    motor=Motor()))
            self.advance(self.traj.t[-1], readout.opacity.animate.set_value(0.35))
            self.end_block()


class MotionRunSeries(RunSeries):
    """Серия прогонов регулятора (ПИД) по подсценам: прогон + докатывание после motors(0, 0) — MOTION секунд видео.
    Потерявший линию робот едет дальше. Между прогонами робота возвращают «рукой» (hand_to), кривые гаснут
    и подпись меняется в это же время — сверх MOTION.

        runs = [(dict(kp=0.1, kd=1.0, ki=0.002), "подпись"), ...]   # одно число — только P (kd = 0)
        runs = [(..., "подпись", "новая формула", [цвета частей])]  # третий и четвертый — смена формулы
    """
    t_on = 0.4            # строка ШИМ снова яркая

    def controller(self, ctrl):
        kw = {"kp": ctrl, "kd": 0.0} if np.isscalar(ctrl) else dict(ctrl)
        extra = {k: kw.pop(k) for k in ("ki",) if k in kw}
        return Controller(**kw), extra

    def simulate(self, pose, ctrl, t_end):
        c, extra = self.controller(ctrl)
        return model.cached_simulate(self.track, tuple(float(v) for v in pose), ctrl=c, t_end=round(t_end, 3),
                                     sensor=self.sensor, motor=Motor(), lost_dist=np.inf, **extra)

    def brake(self, run):
        return model.brake(self.track, run, geom=self.geom, sensor=self.sensor, motor=Motor())

    def plan(self):
        """Прогоны всех подсцен заранее: шкала времени графика — по самому длинному."""
        return [model.fit_run(lambda t, i=i: self.simulate(self.start(i), run[0], t), self.brake,
                              MOTION * self.speed)[0] for i, run in enumerate(self.runs)]

    def panel(self, t_max):
        return GraphPanel(self.fr, self.st, self.phrase, self.runs[0][1], self.y, t_max, np.arange(0, t_max, 0.5),
                          self.signals, readout=dict(signs=model.wheel_signs(Motor()), law=self.law),
                          formula=self.formula, formula_colors=self.formula_colors)

    def construct(self):
        self.track = self.make_track()
        runs = self.plan()
        panel = self.panel(max(r.t[-1] for r in runs) * 1.06)
        self.show(self.track, runs[0], items=[panel.readout])
        self.add(panel)
        self.play(panel.intro(), run_time=0.8)
        readout, curves = panel.readout, None
        for i, ((ctrl, caption, *text), run) in enumerate(zip(self.runs, runs)):   # прогон — подсцена
            if i > 0:
                self.hand_to(self.start(i), FadeOut(curves), panel.set_text(*text, caption=caption))
                self.switch(run)
                self.play(readout.opacity.animate.set_value(1.0), run_time=self.t_on)
            self.begin_segment()
            clock = RunClock(self.t)
            curves = panel.curves(self.traj, clock)
            self.add(curves)
            self.advance(self.traj.t[-1])
            clock.stop()
            self.switch(self.brake(self.traj))
            self.advance(self.traj.t[-1], readout.opacity.animate.set_value(0.35))
            self.end_segment()


class RunClock:
    """Время для кривых прогона: идет за трекером t, после stop() замирает (дальше t перезапускается)."""

    def __init__(self, t):
        self.t, self.live, self.value = t, True, 0.0

    def stop(self):
        self.live = False

    def __call__(self):
        if self.live:
            self.value = self.t.get_value()
        return self.value


QUALITIES = {  # имя → настройки Manim; рендер в любом качестве дает одинаковую картинку, меняется только резкость
    "240p": dict(pixel_width=426, pixel_height=240, frame_rate=15),  # быстрый предпросмотр
    "480p": dict(quality="low_quality"),  # 854×480, 15 к/с
    "720p": dict(quality="medium_quality"),  # 1280×720, 30 к/с
    "1080p": dict(quality="high_quality"),  # 1920×1080, 60 к/с
    "4k": dict(quality="fourk_quality"),  # 3840×2160, 60 к/с
}


def render(scene_cls, quality="480p", still=False, preview=True):
    """Запуск кнопкой ▶ в PyCharm.

    quality: "240p", "480p" (черновик), "720p", "1080p", "4k".
    Видео всех сцен — в одной папке videos/: <Сцена>_<качество>.mp4 (например Scene2c_4k.mp4).
    still=True — только последний кадр: renders/<Сцена>_<качество>.png.
    """
    quality = {"low_quality": "480p", "medium_quality": "720p", "high_quality": "1080p",
               "fourk_quality": "4k"}.get(quality, quality)  # старые имена Manim — тоже можно
    name = scene_cls.__name__
    cfg = {**QUALITIES[quality], "preview": preview, "output_file": f"{name}_{quality}",
           "media_dir": str(ROOT / "media"), "video_dir": str(ROOT / "videos"),
           "partial_movie_dir": str(ROOT / "media" / "partial" / name)}
    if still:
        cfg.update(save_last_frame=True, write_to_movie=False, images_dir=str(ROOT / "renders"))
    with tempconfig(cfg):
        scene_cls().render()
