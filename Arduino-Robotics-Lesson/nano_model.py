"""Физика нового курса: олимпиадная модель (model.py) + И-составляющая урока 3. Без Manim.

Регулятор — как в скетче (шаг delay(10), dt в коде нет):

    integral = integral + err;
    float control = err * kp + (err - errOld) * kd + integral * ki;
    motors(speed + control, speed - control);

Датчики, моторы, кинематика — функции model.py без изменений. Траектория — model.Trajectory; добавлены поля
по тикам: uI = integral · ki и integral.
"""
import hashlib
import inspect
from collections import deque
from pathlib import Path

import numpy as np

import model
from model import Trajectory
from params import Controller, Geometry, Motor

from nano_track import SENSOR

EXTRA = ("uI", "integral")
CACHE = model.CACHE
_SOURCE = hashlib.sha1(Path(__file__).read_bytes() + Path(model.__file__).read_bytes()).hexdigest()


def simulate(track, pose0, ctrl=Controller(), ki=0.0, t_end=5.0, geom=Geometry(), sensor=SENSOR, motor=Motor(),
             lost_dist=100.0, stop_below=None):
    """Как model.simulate, плюс И: integral = integral + err, control = uP + uD + integral·ki.
    lost_dist = np.inf — робот, потерявший линию, едет дальше (сцена подбора P)."""
    s = np.array([*pose0, 0.0, 0.0, 0.0, 0.0])
    moving = [False, False]
    cont, ticks, extra = [], [], []
    e_old, integral, lost_at = None, 0.0, np.nan
    lag = deque([(0.0, 0.0)] * int(round(motor.delay / ctrl.period)))
    for k in range(int(round(t_end / ctrl.period))):
        t0 = k * ctrl.period
        L, R = model.to_channels(model.read_sensors(track, s[:3], geom, sensor), sensor)
        if stop_below is not None and L < stop_below and R < stop_below:
            break
        e = model.error(L, R, ctrl)
        e_old = e if e_old is None else e_old
        integral = integral + e
        uP, uD = model.control_parts(e, e_old, ctrl)
        uI = integral * ki
        pL, pR = model.wheel_pwm(uP + uD + uI, ctrl, motor)
        e_old = e
        s, moving, chunk = model._wheel_step(s, moving, model.delayed(lag, (pL, pR)), t0, ctrl.period, motor, geom)
        cont.append(chunk)
        ticks.append((t0, L, R, e, uP, uD, pL, pR))
        extra.append((uI, integral))
        if np.isfinite(lost_dist) and min(d.min() for d in track.distances(s[None, :2])) > lost_dist:
            lost_at = t0 + ctrl.period
            break
    traj = model._pack(np.hstack(cont), np.array(ticks), lost_at)
    traj.uI, traj.integral = np.array(extra).T
    return traj


def cached_simulate(track, pose0, **kw):
    """simulate() с кэшем в .cache/ (ключ — параметры и текст nano_model.py и model.py)."""
    args = inspect.signature(simulate).bind(track, pose0, **kw)
    args.apply_defaults()
    rest = {k: v for k, v in args.arguments.items() if k not in ("track", "pose0")}
    key = repr(["nano", _SOURCE, model._fingerprint(track), tuple(map(float, pose0)),
                sorted((k, repr(v)) for k, v in rest.items())])
    path = CACHE / ("n" + hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz")
    if path.exists():
        z = np.load(path)
        traj = Trajectory.load(path)
        for k in EXTRA:
            setattr(traj, k, z[k])
        return traj
    traj = simulate(track, pose0, **kw)
    CACHE.mkdir(exist_ok=True)
    np.savez(path, **{k: getattr(traj, k) for k in (*Trajectory.__dataclass_fields__, *EXTRA)})
    return traj
