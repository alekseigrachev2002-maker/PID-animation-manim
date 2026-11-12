import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Geometry, Motor, Sensor  # noqa: E402

G, C = Geometry(), Controller()
S = Sensor()                                            # в модели датчики одинаковые (среднее замера)
S_REAL = Sensor(white=(920, 958), black=(191, 216))     # замер двух случайных датчиков — немного разные
LINE = track.Track(track.Route(start=(0, -300), heading="вверх").forward(3300))


def e_at(x=0.0, psi_deg=0.0):
    L, R = model.read_sensors(LINE, (x, 0.0, np.radians(psi_deg)), G, S)
    return model.error(L, R, C)


def pwm_at(x=0.0, psi_deg=0.0):
    return model.wheel_pwm(C.kp * e_at(x, psi_deg), C, Motor())


def test_centered_error_zero():
    assert e_at() == pytest.approx(0.0, abs=1e-9)


def test_centered_error_is_white_mismatch():
    """Реальные датчики, робот по центру, оба на белом: err ≈ разница на белом (920 − 958 = −38)."""
    L, R = model.read_sensors(LINE, (0.0, 0.0, 0.0), G, S_REAL)
    assert model.error(L, R, C) == pytest.approx(S_REAL.white[0] - S_REAL.white[1], abs=1.0)


@pytest.mark.parametrize("psi", [9, 12, 15])
def test_turned_left_steers_right(psi):
    pL, pR = pwm_at(psi_deg=psi)
    assert pL > pR


@pytest.mark.parametrize("psi", [-9, -12, -15])
def test_turned_right_steers_left(psi):
    pL, pR = pwm_at(psi_deg=psi)
    assert pL < pR


def test_shifted_right_steers_left():
    pL, pR = pwm_at(x=20)
    assert pL < pR


def test_turned_left_heading_decreases():
    traj = model.simulate(LINE, (0.0, 0.0, np.radians(12)), t_end=0.1)
    assert traj.psi[-1] < traj.psi[0]


# Мёртвая зона из геометрии: оба датчика на белом, пока до края линии больше 3σ.
# Смещение на x: датчик в sensor_x − x от осевой. Поворот на ψ: боковое положение датчика
# sensor_x·cosψ − sensor_y·sinψ = r·sin(α − ψ), где r — расстояние до датчика, α = atan2(sensor_x, sensor_y).
DZ_X = G.sensor_x - S.line_width / 2 - 3 * S.sigma
_r = np.hypot(G.sensor_x, G.sensor_y)
DZ_PSI = np.degrees(np.arctan2(G.sensor_x, G.sensor_y) - np.arcsin((S.line_width / 2 + 3 * S.sigma) / _r))


@pytest.mark.parametrize("k", np.linspace(-0.95, 0.95, 21))
def test_dead_zone_offset(k):
    assert abs(e_at(x=k * DZ_X)) < 1


@pytest.mark.parametrize("k", np.linspace(-0.95, 0.95, 21))
def test_dead_zone_heading(k):
    assert abs(e_at(psi_deg=k * DZ_PSI)) < 1


@pytest.mark.parametrize("sign", [1, -1])
def test_outside_dead_zone(sign):
    edge_x = G.sensor_x - S.line_width / 2          # датчик ровно на краю линии
    assert abs(e_at(x=sign * (edge_x - S.sigma))) > 100
    assert abs(e_at(psi_deg=sign * (DZ_PSI + 4))) > 100


def test_dead_zone_numbers_for_field_line():
    """Линия поля 30 мм: мёртвая зона узкая — около ±1.5 мм и ±0.8°."""
    assert S.line_width == 30
    assert DZ_X == pytest.approx(1.29, abs=0.05) and DZ_PSI == pytest.approx(0.8, abs=0.1)


def test_stuck_wheel_needs_start_threshold():
    m = Motor()
    assert not model.update_moving(False, 0.0, 60, m)
    assert model.update_moving(False, 0.0, 70, m)
    assert model.update_moving(True, 50.0, 55, m)


def test_line_distance_exact_between_samples():
    d = LINE.lines[0].distance(np.array([[3.0, 0.25], [-7.0, 10.1]]))
    assert d == pytest.approx([3.0, 7.0], abs=1e-9)


def test_delay_queue():
    from collections import deque
    lag = deque([(0, 0)] * 2)
    out = [model.delayed(lag, (k, k)) for k in (1, 2, 3)]
    assert out == [(0, 0), (0, 0), (1, 1)]
    assert model.delayed(deque(), (5, 5)) == (5, 5)


def test_swapped_wiring_steers_away():
    """Белое выше, motors(speed + u, speed − u): если первый аргумент крутит правое колесо — рулит от линии."""
    traj = model.simulate(LINE, (0.0, 0.0, np.radians(12)), t_end=0.1, motor=Motor(swapped=True))
    assert traj.psi[-1] > traj.psi[0]


def test_cache_key_sees_defaults(tmp_path, monkeypatch):
    """Ключ кэша строится по всем параметрам: явный Motor() и умолчание дают один файл, другой Motor — второй."""
    monkeypatch.setattr(model, "CACHE", tmp_path)
    model.cached_simulate(LINE, (0.0, 0.0, 0.2), t_end=0.05)
    model.cached_simulate(LINE, (0.0, 0.0, 0.2), t_end=0.05, motor=Motor())
    assert len(list(tmp_path.glob("*.npz"))) == 1
    model.cached_simulate(LINE, (0.0, 0.0, 0.2), t_end=0.05, motor=Motor(swapped=True))
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_line_side():
    ln = LINE.lines[0]   # вертикальная линия, ход вдоль +y: слева — x < 0
    assert list(ln.side(np.array([[-5.0, 100.0], [5.0, 100.0]]))) == [1, -1]


@pytest.mark.parametrize("x, psi, lost", [(0, 0, False), (8, 5, False), (-100, 15, True), (60, 0, True),
                                          (0, 15, False)])
def test_off_line(x, psi, lost):
    assert model.off_line(LINE, (x, 0.0, np.radians(psi))) == lost


def test_move_without_roll_keeps_wheels():
    tr = model.move(LINE, (-100.0, 50.0, 0.3), (0.0, 50.0, 0.3), 1.0, roll=False)
    assert tr.x[0] == pytest.approx(-100) and tr.x[-1] == pytest.approx(0)
    assert np.all(tr.sL == 0) and np.all(tr.sR == 0)


@pytest.mark.parametrize("s_swap, m_swap, toward", [(False, False, True), (True, True, True),
                                                   (False, True, False), (True, False, False)])
def test_wiring_combinations(s_swap, m_swap, toward):
    """Белое выше, motors(speed + u, speed − u): к линии — если не переставлено ничего или переставлено и то, и другое."""
    traj = model.simulate(LINE, (0.0, 0.0, np.radians(12)), t_end=0.1,
                          sensor=Sensor(swapped=s_swap), motor=Motor(swapped=m_swap))
    assert (traj.psi[-1] < traj.psi[0]) == toward


def test_channels_roundtrip():
    phys = np.array([300.0, 700.0])
    a0, a1 = model.to_channels(phys, Sensor(swapped=True))
    assert (a0, a1) == (700.0, 300.0)
    assert tuple(model.to_channels(np.array([a0, a1]), Sensor(swapped=True))) == (300.0, 700.0)


def test_wheel_signs_match_pwm():
    for sw in (False, True):
        m = Motor(swapped=sw)
        sl, sr = model.wheel_signs(m)
        pl, pr = model.wheel_pwm(40.0, C, m)
        assert (pl, pr) == (C.speed + sl * 40.0, C.speed + sr * 40.0)


def test_path_right_turn_geometry():
    tr = track.Track(track.Route(start=(0, -100), heading="вверх").forward(100).mark("поворот")
                     .right(R=100).forward(50))
    ln = tr.lines[0]
    assert ln.points[-1] == pytest.approx([150, 100], abs=1e-6)       # дуга до (100, 100), затем 50 вправо
    x, y, psi = ln.pose_at(ln.s[-1] - 1)
    assert np.degrees(psi) == pytest.approx(-90, abs=1e-6)            # курс «вправо»
    assert tr.pose("поворот") == pytest.approx((0, 0, 0), abs=1e-6)
    assert tr.pose("поворот", -40) == pytest.approx((0, -40, 0), abs=1e-6)


def test_path_left_turn_radius():
    ln = track.Track(track.Route(heading="вверх").left(R=80)).lines[0]
    assert np.hypot(*(ln.points - [-80, 0]).T) == pytest.approx(np.full(len(ln.points), 80), abs=1e-6)
    assert ln.points[-1] == pytest.approx([-80, 80], abs=1e-6)


def test_start_from_rest_accelerates():
    traj = model.simulate(LINE, (0.0, 0.0, 0.0), t_end=0.5)
    v = (traj.vL + traj.vR) / 2
    assert v[0] == 0 and v[-1] == pytest.approx(Motor().v_at_150, rel=0.02)
    assert np.all(np.diff(v) >= -1e-9)                       # разгон без скачков


def test_brake_stops_smoothly():
    run = model.simulate(LINE, (0.0, 0.0, 0.0), t_end=0.5)
    stop = model.brake(LINE, run)
    v = (stop.vL + stop.vR) / 2
    assert v[0] == pytest.approx((run.vL[-1] + run.vR[-1]) / 2, rel=1e-3)   # без скачка скорости
    assert v[-1] < 2 and stop.y[-1] > run.y[-1]                               # докатился и встал
    assert stop.sL[0] == pytest.approx(run.sL[-1])


def test_drive_below_start_threshold_stays():
    tr = model.drive(LINE, (0.0, 0.0, 0.0), (60, 60), 0.5)          # ШИМ 60 < порога трогания 67
    assert tr.y[-1] == pytest.approx(0.0, abs=1e-9)


def test_drive_spin_in_place_keeps_axle():
    tr = model.drive(LINE, (0.0, 0.0, 0.0), (-150, 150), 1.0)       # левое назад, правое вперёд
    assert np.hypot(tr.x, tr.y).max() < 1e-6
    assert tr.psi[-1] > 1.0                                           # повернулся влево (против часовой)


def test_drive_arc_radius():
    m = Motor()
    tr = model.drive(LINE, (0.0, 0.0, 0.0), (200, 100), 3.0)
    vl, vr = m.k * (200 - m.pwm_stop), m.k * (100 - m.pwm_stop)
    R = Geometry().track_width / 2 * (vl + vr) / (vl - vr)            # радиус дуги середины оси
    v, w = tr.speed_at(2.5)
    assert v / abs(w) == pytest.approx(R, rel=0.01)


def test_brake_until_still_with_slow_motor():
    slow = Motor(tau=0.3)
    run = model.simulate(LINE, (0.0, 0.0, 0.0), t_end=1.5, motor=slow)
    stop = model.brake(LINE, run, motor=slow)
    assert stop.t[-1] > 0.8                                           # раньше обрезалось на 0.8 с
    assert stop.speed_at(stop.t[-1])[0] < 5


def test_speed_at_matches_wheels():
    run = model.simulate(LINE, (0.0, 0.0, 0.0), t_end=1.0, sensor=S)
    v, w = run.speed_at(0.9)
    assert v == pytest.approx((run.vL[-1] + run.vR[-1]) / 2, rel=0.01) and abs(w) < 1e-3


def test_drive_ramp_then_stop():
    ramp = lambda t: (255 * t / 0.6,) * 2
    tr = model.drive(LINE, (0.0, 0.0, 0.0), ramp, 0.6, stop=True)
    i67 = np.argmax(tr.pwmL > Motor().pwm_start)
    assert tr.pwmL.max() == pytest.approx(255 * (0.6 - 0.005) / 0.6, rel=0.01)   # команда — лесенка по тактам
    assert tr.y[np.searchsorted(tr.t, tr.tt[i67])] == pytest.approx(0, abs=1e-6)  # до порога трогания стоит
    assert tr.pwmL[-1] == 0 and tr.speed_at(tr.t[-1])[0] < 5                     # потом стоп и докатился


# ---------- перекрёсток (сцена 3) ----------

CROSS = track.Track(track.Route(start=(0, -100), heading="вверх").forward(600),
                    track.Route(start=(-150, 300), heading="вправо").forward(300))


def test_stop_below_ends_on_crossing():
    """Траектория кончается на первом тике, где L и R оба ниже порога; до этого — нет."""
    run = model.simulate(CROSS, (0.0, 0.0, 0.0), t_end=3.0, stop_below=300)
    assert run.t[-1] < 2.9
    assert np.all((run.L >= 300) | (run.R >= 300))
    L, R = model.read_sensors(CROSS, (run.x[-1], run.y[-1], run.psi[-1]), G, S)
    assert max(L, R) < 300
    assert abs(run.y[-1] + G.sensor_y - 300) < 15           # датчики на полосе перекрёстка


def test_brake_waits_for_delay():
    """Стоп доходит до колёс через Motor.delay: первые delay секунд робот едет, как ехал."""
    m = Motor(delay=0.02)
    run = model.simulate(LINE, (0.0, 0.0, 0.0), t_end=0.5, motor=m)
    stop = model.brake(LINE, run, motor=m)
    v = (stop.vL + stop.vR) / 2
    k = np.searchsorted(stop.t, 0.02)
    assert v[k] == pytest.approx(v[0], rel=0.02)
    assert stop.y[-1] - stop.y[0] > model.brake(LINE, run).y[-1] - run.y[-1] + 3   # прокат длиннее, чем без задержки


def test_chain_continues_time_and_wheel_path():
    a = model.drive(LINE, (0.0, 0.0, 0.0), (150, 150), 0.3, stop=True)
    b = model.drive(LINE, (a.x[-1], a.y[-1], a.psi[-1]), (150, 150), 0.3, stop=True)
    ab = model.chain(a, b)
    assert ab.t[-1] == pytest.approx(a.t[-1] + b.t[-1])
    assert np.all(np.diff(ab.t) > 0)
    assert ab.sL[-1] == pytest.approx(a.sL[-1] + b.sL[-1] - b.sL[0])
    assert len(ab.tt) == len(a.tt) + len(b.tt)
