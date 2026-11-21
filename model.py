"""Физика робота: датчики, моторы, кинематика, регулятор. Без Manim.

Поза: (x, y, psi), psi — курс от +y против часовой (psi > 0 — повернут влево).
Регулятор дискретный (ZOH между тиками), робот непрерывный: solve_ivp от тика до тика.
Регулятор — ПИД, как в скетчах (шаг delay(10), dt в коде нет): integral = integral + err,
control = kp·err + kd·(err − errOld) + ki·integral; ki = 0 — ПД олимпиадного курса.
Дальномер (урок 4 нового курса): sonar_distance — расстояние до кубиков трассы в луче.
"""
import hashlib
import inspect
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf

from params import Controller, Geometry, Motor, Sensor, Sonar

GRID = 0.001          # шаг непрерывной траектории, с
TICK_FIELDS = ("tt", "L", "R", "e", "uP", "uD", "pwmL", "pwmR", "uI", "integral")   # поля Trajectory по тикам
CACHE = Path(__file__).parent / ".cache"


# ---------- датчики ----------

def body_to_world(pose, pts):
    """Точки из системы робота (мм) в мировую. pose = (x, y, psi)."""
    x, y, psi = pose
    c, s = np.cos(psi), np.sin(psi)
    pts = np.asarray(pts, float)
    return np.column_stack([x + c * pts[:, 0] - s * pts[:, 1],
                            y + s * pts[:, 0] + c * pts[:, 1]])


def sensor_points(pose, geom: Geometry):
    """Оптические центры [L, R] в мировых координатах."""
    body = [[-geom.sensor_x, geom.sensor_y], [geom.sensor_x, geom.sensor_y]]
    return body_to_world(pose, body)


def strip_coverage(d, width, sigma):
    """Доля гауссова пятна, попавшая на полосу ширины width; d — расстояние до ее оси."""
    k = 1 / (sigma * np.sqrt(2))
    return 0.5 * (erf((width / 2 - d) * k) + erf((width / 2 + d) * k))


def coverage(track, pts, sensor: Sensor):
    """Покрытие с учетом всех линий трассы: 1 − Π(1 − c_k)."""
    free = np.ones(len(pts))
    for d in track.distances(pts):
        free *= 1 - strip_coverage(d, sensor.line_width, sensor.sigma)
    return 1 - free


def adc(cov, sensor: Sensor, rng=None):
    """Показания АЦП [L, R] по покрытию."""
    W, B = np.asarray(sensor.white, float), np.asarray(sensor.black, float)
    val = W + (B - W) * cov
    if sensor.noise > 0 and rng is not None:
        val = val + rng.normal(0, sensor.noise, size=val.shape)
    return val


def read_sensors(track, pose, geom, sensor, rng=None):
    """Показания физических датчиков [левый, правый]."""
    return adc(coverage(track, sensor_points(pose, geom), sensor), sensor, rng)


def to_channels(phys, sensor: Sensor):
    """Физические [левый, правый] → то, что видит код: (A0 = L, A1 = R). Обратное преобразование то же."""
    left, right = phys[..., 0], phys[..., 1]
    return (right, left) if sensor.swapped else (left, right)


def channel_sides(sensor: Sensor):
    """Какой физический датчик стоит на A0 и на A1."""
    return ("правый", "левый") if sensor.swapped else ("левый", "правый")


# ---------- регулятор (порт followLine) ----------

def error(L, R, ctrl: Controller):
    return L * ctrl.balance - R


def control_parts(e, e_old, ctrl: Controller):
    """(uP, uD) = (kp·e, kd·(e − e_old))."""
    return ctrl.kp * e, ctrl.kd * (e - e_old)


def motor_pwm(u, ctrl: Controller):
    """motors(speed + u, speed − u) с constrain ±255 → (первый аргумент, второй) — как в скетче урока 2."""
    lim = ctrl.pwm_limit
    return np.clip(ctrl.speed + u, -lim, lim), np.clip(ctrl.speed - u, -lim, lim)


def wheel_signs(motor: Motor):
    """Знаки u в скорости физических колес: ШИМ левого = speed + sL·u, правого = speed + sR·u."""
    return (-1, 1) if motor.swapped else (1, -1)


def wheel_pwm(u, ctrl: Controller, motor: Motor):
    """ШИМ на физических колесах (левое, правое) с учетом того, как моторы подключены к IN1…IN4."""
    a, b = motor_pwm(u, ctrl)
    return (b, a) if motor.swapped else (a, b)


# ---------- моторы ----------

def update_moving(moving, v, pwm, motor: Motor):
    """Гистерезис трения: стоящее колесо трогается при |ШИМ| > pwm_start,
    крутящееся считается вставшим, когда |v| < stop_eps и |ШИМ| < pwm_start."""
    if moving:
        return not (abs(v) < motor.stop_eps and abs(pwm) < motor.pwm_start)
    return abs(pwm) > motor.pwm_start


def wheel_target(pwm, moving, motor: Motor):
    """Установившаяся скорость колеса, мм/с."""
    return motor.speed(pwm) if moving else 0.0


def turn_targets(vt, motor: Motor):
    """Скорости, к которым тянутся колеса: от разницы целевых до поворота доходит доля Motor.turn_share."""
    mid, half = (vt[0] + vt[1]) / 2, (vt[0] - vt[1]) / 2 * motor.turn_share
    return mid + half, mid - half


def delayed(lag, cmd):
    """Очередь задержки: кладем новую команду, достаем самую старую (пустая очередь — без задержки)."""
    lag.append(cmd)
    return lag.popleft()


# ---------- кинематика ----------

def rhs(t, s, vtL, vtR, tau, b):
    """s = [x, y, psi, vL, vR, sL, sR]."""
    _, _, psi, vL, vR, _, _ = s
    v = (vL + vR) / 2
    return [-v * np.sin(psi), v * np.cos(psi), (vR - vL) / b,
            (vtL - vL) / tau, (vtR - vR) / tau, vL, vR]


# ---------- траектория ----------

@dataclass
class Trajectory:
    # непрерывная часть, шаг GRID
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    vL: np.ndarray
    vR: np.ndarray
    sL: np.ndarray       # путь колеса, мм (фаза протектора = s / R)
    sR: np.ndarray
    # по тикам регулятора
    tt: np.ndarray
    L: np.ndarray
    R: np.ndarray
    e: np.ndarray
    uP: np.ndarray
    uD: np.ndarray
    pwmL: np.ndarray
    pwmR: np.ndarray
    uI: np.ndarray       # ki · integral
    integral: np.ndarray # сумма err
    lost_at: float = np.nan   # время схода с линии (nan — не сошел)

    def at(self, t):
        """Поза и скорости в момент t (линейная интерполяция по сетке GRID)."""
        return {k: np.interp(t, self.t, getattr(self, k))
                for k in ("x", "y", "psi", "vL", "vR", "sL", "sR")}

    def speed_at(self, t, h=GRID):
        """Скорость (мм/с) и угловая скорость (рад/с) робота в момент t — по изменению позы."""
        a, b = max(t - h, self.t[0]), min(t + h, self.t[-1])
        if b <= a:
            return 0.0, 0.0
        pa, pb = self.at(a), self.at(b)
        return (float(np.hypot(pb["x"] - pa["x"], pb["y"] - pa["y"]) / (b - a)),
                float((pb["psi"] - pa["psi"]) / (b - a)))

    def signals_at(self, t):
        """Значения регулятора, действующие в момент t (ZOH)."""
        i = np.clip(np.searchsorted(self.tt, t, side="right") - 1, 0, len(self.tt) - 1)
        return {k: getattr(self, k)[i] for k in TICK_FIELDS[1:]}

    def save(self, path):
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(**{f.name: (z[f.name] if z[f.name].ndim else float(z[f.name]))
                      for f in fields(cls)})


def _wheel_step(s, moving, cmd, t0, dt, motor: Motor, geom: Geometry):
    """Один такт с командами ШИМ cmd = (левое, правое): обновить «стоит/крутится» и проинтегрировать движение.
    Возвращает (состояние в конце такта, флаги, кусок непрерывной траектории с шагом GRID)."""
    moving = [update_moving(m, v, p, motor) for m, v, p in zip(moving, s[3:5], cmd)]
    s = s.copy()
    s[3] = s[3] if moving[0] else 0.0
    s[4] = s[4] if moving[1] else 0.0
    vt = turn_targets((wheel_target(cmd[0], moving[0], motor), wheel_target(cmd[1], moving[1], motor)), motor)
    vt = tuple(v if mv else 0.0 for v, mv in zip(vt, moving))      # стоящее колесо стоит
    sol = solve_ivp(rhs, (t0, t0 + dt), s, args=(*vt, motor.tau, geom.track_width),
                    dense_output=True, rtol=1e-8, atol=1e-8)
    tg = t0 + GRID * np.arange(int(round(dt / GRID)))
    return sol.y[:, -1].copy(), moving, np.vstack([tg, sol.sol(tg)])


def simulate(track, pose0, ctrl=Controller(), t_end=5.0, geom=Geometry(), sensor=Sensor(),
             motor=Motor(), v0=0.0, seed=0, lost_dist=100.0, stop_below=None, ki=0.0, stop_at=1):
    """Езда по трассе регулятором ПИД (ki = 0 — ПД). v0 — начальная скорость колес (по умолчанию робот стоит).

    Останавливается, если середина оси дальше lost_dist от всех линий (lost_dist = np.inf — потерявший линию
    робот едет дальше).
    stop_below — порог перекрестка: L и R оба ниже — перекресток; счет по фронту, как в уроке 3. На stop_at-м
    перекрестке регулятор не вызывается и траектория кончается (как в скетче: if (перекресток) motors(0, 0);
    else followLine();) — дальше brake(); прошлые перекрестки робот проезжает. Перекресток, на котором робот
    стоит на старте, не считается.
    """
    rng = np.random.default_rng(seed)
    s = np.array([*pose0, v0, v0, 0.0, 0.0])
    moving = [v0 != 0, v0 != 0]
    cont, ticks = [], []
    e_old, integral, lost_at = None, 0.0, np.nan
    count, was = 0, None
    # команды, отданные до старта: стоящему роботу — нули, едущему — прямо
    prior = (ctrl.speed, ctrl.speed) if v0 else (0.0, 0.0)
    lag = deque([prior] * int(round(motor.delay / ctrl.period)))
    for k in range(int(round(t_end / ctrl.period))):
        t0 = k * ctrl.period
        L, R = to_channels(read_sensors(track, s[:3], geom, sensor, rng), sensor)
        if stop_below is not None:
            cross = L < stop_below and R < stop_below
            was = cross if was is None else was          # стоит на перекрестке на старте — он не в счет
            count += cross and not was
            was = cross
            if cross and count >= stop_at:
                break
        e = error(L, R, ctrl)
        e_old = e if e_old is None else e_old   # без рывка D на первом тике
        integral = integral + e
        uP, uD = control_parts(e, e_old, ctrl)
        uI = ki * integral
        pL, pR = wheel_pwm(uP + uD + uI, ctrl, motor)
        e_old = e
        s, moving, chunk = _wheel_step(s, moving, delayed(lag, (pL, pR)), t0, ctrl.period, motor, geom)
        cont.append(chunk)
        ticks.append((t0, L, R, e, uP, uD, pL, pR, uI, integral))
        if np.isfinite(lost_dist) and min(d.min() for d in track.distances(s[None, :2])) > lost_dist:
            lost_at = t0 + ctrl.period
            break
    return _pack(np.hstack(cont), np.array(ticks), lost_at)


def _commanded(track, s, moving, cmd, t_on, ctrl, geom, sensor, motor, stop=False, t_guard=30.0, pending=None):
    """Езда без регулятора: команды ШИМ колесам cmd(t) → (левое, правое) пока t < t_on.

    stop: затем motors(0, 0), пока оба колеса не встанут — в той же траектории (график видит и стоп).
    pending — команды, еще идущие по очереди задержки; по умолчанию нули (робот стоял):
    новые команды начинают действовать через Motor.delay.
    """
    cont, ticks = [], []
    n = int(round(motor.delay / ctrl.period))
    lag = deque(pending if pending is not None else [(0.0, 0.0)] * n)
    k = 0
    while True:
        t0 = k * ctrl.period
        on = t0 < t_on
        if not on and not stop:
            break
        c = tuple(map(float, cmd(t0))) if on else (0.0, 0.0)
        L, R = to_channels(read_sensors(track, s[:3], geom, sensor), sensor)
        s, moving, chunk = _wheel_step(s, moving, delayed(lag, c), t0, ctrl.period, motor, geom)
        cont.append(chunk)
        ticks.append((t0, L, R, error(L, R, ctrl), 0.0, 0.0, *c, 0.0, 0.0))
        if not on and not any(moving):
            break
        if t0 > t_on + t_guard:
            raise RuntimeError(f"колеса не встали за {t_guard} с — проверьте Motor.tau и stop_eps")
        k += 1
    return _pack(np.hstack(cont), np.array(ticks), np.nan)


def brake(track, traj, ctrl=Controller(), geom=Geometry(), sensor=Sensor(), motor=Motor()):
    """Остановка после прогона: motors(0, 0), робот докатывается, пока колеса не встанут.

    Стартует из конца траектории traj с теми же скоростями колес; путь колес продолжается.
    Последние команды регулятора еще в очереди задержки — колеса узнают о стопе через Motor.delay.
    """
    s = np.array([traj.x[-1], traj.y[-1], traj.psi[-1], traj.vL[-1], traj.vR[-1], traj.sL[-1], traj.sR[-1]])
    moving = [abs(s[3]) > motor.stop_eps, abs(s[4]) > motor.stop_eps]
    n = int(round(motor.delay / ctrl.period))
    pending = list(zip(traj.pwmL, traj.pwmR))[len(traj.tt) - n:] if n else []
    return _commanded(track, s, moving, lambda t: (0, 0), 0.0, ctrl, geom, sensor, motor, stop=True,
                      pending=pending)


def drive(track, pose0, pwm, t_on, stop=False, ctrl=Controller(), geom=Geometry(), sensor=Sensor(), motor=Motor()):
    """Езда с места без регулятора: ШИМ физических колес pwm = (левое, правое) или функция pwm(t) → (лев, прав).

    Разгон, порог трогания, отставание — по физике мотора. stop: после t_on — motors(0, 0) и докатывание.
    """
    fn = pwm if callable(pwm) else (lambda t: pwm)
    s = np.array([*pose0, 0.0, 0.0, 0.0, 0.0])
    return _commanded(track, s, [False, False], fn, t_on, ctrl, geom, sensor, motor, stop=stop)


def off_line(track, pose, geom=Geometry(), sensor=Sensor(), seen=0.02):
    """Робот потерял линию: ни один датчик ее не видит и оба лежат по одну сторону от нее."""
    pts = sensor_points(pose, geom)
    if coverage(track, pts, sensor).max() > seen:
        return False
    return all(ln.side(pts)[0] == ln.side(pts)[1] for ln in track.lines)


def smoothstep(s):
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def move(track, pose_a, pose_b, duration, roll=True, dt=0.005, geom=Geometry(), sensor=Sensor()):
    """Плавный переход позы a → b за duration (возврат между прогонами). roll=False — колеса не крутятся
    (робота переставили / линия подъехала)."""
    t = np.arange(0, duration + dt / 2, dt)
    s = smoothstep(t / duration)
    a, b = np.asarray(pose_a, float), np.asarray(pose_b, float)
    x, y, psi = (a[k] + (b[k] - a[k]) * s for k in range(3))
    traj = prescribed(track, t, x, y, psi, geom, sensor)
    if not roll:
        traj.sL[:] = 0.0
        traj.sR[:] = 0.0
    return traj


def prescribed(track, t, x, y, psi, geom=Geometry(), sensor=Sensor(), ctrl=Controller(), seed=0):
    """Движение, заданное руками (сцена 1): датчики через ту же функцию, регулятор не действует."""
    rng = np.random.default_rng(seed)
    t, x, y, psi = (np.broadcast_to(np.asarray(a, float), np.shape(t)).copy() for a in (t, x, y, psi))
    phys = np.array([read_sensors(track, p, geom, sensor, rng) for p in zip(x, y, psi)])
    LR = np.column_stack(to_channels(phys, sensor))
    e = error(LR[:, 0], LR[:, 1], ctrl)
    # путь колес: смещение по траектории ± поворот вокруг оси
    ds = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    turn = geom.track_width / 2 * (psi - psi[0])
    z = np.zeros_like(t)
    cont = np.vstack([t, x, y, psi, z, z, ds - turn, ds + turn])
    ticks = np.column_stack([t, LR[:, 0], LR[:, 1], e, z, z, z, z, z, z])
    return _pack(cont, ticks, np.nan)


def chain(*trajs):
    """Склеить отрезки в одну траекторию: время и путь колес каждого продолжают предыдущий.

    Для графика через всю сцену (стоит → переставили → едет → стоп). Стык — в покое или run → brake.
    """
    out = {f.name: [] for f in fields(Trajectory) if f.name != "lost_at"}
    t0, s0 = 0.0, np.zeros(2)
    for i, tr in enumerate(trajs):
        skip = 1 if i > 0 else 0                  # первая точка отрезка = последняя предыдущего
        for k in ("x", "y", "psi", "vL", "vR"):
            out[k].append(getattr(tr, k)[skip:])
        out["t"].append(tr.t[skip:] + t0)
        out["sL"].append(tr.sL[skip:] - tr.sL[0] + s0[0])
        out["sR"].append(tr.sR[skip:] - tr.sR[0] + s0[1])
        out["tt"].append(tr.tt + t0)
        for k in TICK_FIELDS[1:]:
            out[k].append(getattr(tr, k))
        t0, s0 = out["t"][-1][-1], np.array([out["sL"][-1][-1], out["sR"][-1][-1]])
    return Trajectory(**{k: np.concatenate(v) for k, v in out.items()})


def _pack(cont, ticks, lost_at):
    c = dict(zip(("t", "x", "y", "psi", "vL", "vR", "sL", "sR"), cont))
    k = dict(zip(TICK_FIELDS, ticks.T))
    return Trajectory(**c, **k, lost_at=lost_at)


# ---------- дальномер ----------

def block_hit(block, origin, heading, half_deg):
    """Расстояние от origin по курсу heading (рад, как ψ) до кубика block (center, side), если его ближний край
    внутри луча ±half_deg; иначе None."""
    d = np.asarray(block.center, float) - np.asarray(origin, float)
    fwd = np.array([-np.sin(heading), np.cos(heading)])
    along = d @ fwd
    if along <= 0:
        return None
    off = abs(d @ np.array([np.cos(heading), np.sin(heading)]))
    if off - block.side / 2 > along * np.tan(np.radians(half_deg)):
        return None
    return float(along - block.side / 2)


def sonar_distance(pose, blocks, sonar=Sonar(), max_mm=600.0):
    """Что покажет дальномер в позе pose = (x, y, ψ): расстояние от излучателей до ближнего кубика (track.blocks)
    в луче ±beam_deg, мм; кубика в луче нет — max_mm."""
    x, y, psi = pose
    o = np.array([x, y]) + np.array([-np.sin(psi), np.cos(psi)]) * sonar.front
    hits = [d for b in blocks if (d := block_hit(b, o, psi, sonar.beam_deg)) is not None]
    return min(hits + [max_mm])


# ---------- подбор ----------

def bisect(done, lo, hi, n=30):
    """Наименьшее x в [lo, hi], при котором done(x) истинно (done монотонна): подбор времени команды, угла."""
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if done(mid) else (mid, hi)
    return (lo + hi) / 2


def fit_run(sim, stop, target):
    """Прогон run = sim(t_end), который вместе с докатыванием stop(run) длится target секунд модели
    (подстройка t_end по перебору/недобору, до 6 шагов). Возвращает (run, stop(run))."""
    t_end = target - 0.25
    for _ in range(6):
        run = sim(round(t_end, 3))
        tail = stop(run)
        over = run.t[-1] + tail.t[-1] - target
        if abs(over) <= 0.01:
            break
        t_end -= over + 0.005
    return run, tail


def end_pose(traj):
    """Последняя поза траектории (x, y, ψ)."""
    return float(traj.x[-1]), float(traj.y[-1]), float(traj.psi[-1])


# ---------- кэш ----------

_SOURCE = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()     # версия физики — часть ключа кэша

def _fingerprint(obj):
    if hasattr(obj, "lines"):
        return [ln.points.tobytes() for ln in obj.lines]
    return repr(obj)


def cached_simulate(track, pose0, **kw):
    """simulate() с кэшем в .cache/<хэш>.npz. Ключ — все параметры (включая значения по умолчанию) и текст
    model.py: поменяли физику — старые траектории в кэше не подхватятся."""
    args = inspect.signature(simulate).bind(track, pose0, **kw)
    args.apply_defaults()
    rest = {k: v for k, v in args.arguments.items() if k not in ("track", "pose0")}
    key = repr([_SOURCE, _fingerprint(track), tuple(pose0), sorted((k, repr(v)) for k, v in rest.items())])
    path = CACHE / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz")
    if path.exists():
        return Trajectory.load(path)
    traj = simulate(track, pose0, **kw)
    CACHE.mkdir(exist_ok=True)
    traj.save(path)
    return traj
