import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Geometry, Motor, Sensor  # noqa: E402

G, S, C = Geometry(), Sensor(), Controller()
LINE = track.Track(track.Route(start=(0, -300), heading="вверх").forward(3300))


def e_at(x=0.0, psi_deg=0.0):
    L, R = model.read_sensors(LINE, (x, 0.0, np.radians(psi_deg)), G, S)
    return model.error(L, R, C)


def pwm_at(x=0.0, psi_deg=0.0):
    return model.wheel_pwm(C.kp * e_at(x, psi_deg), C, Motor())


def test_centered_error_zero():
    assert e_at() == pytest.approx(0.0, abs=1e-9)


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


@pytest.mark.parametrize("x", np.linspace(-11, 11, 23))
def test_dead_zone_offset(x):
    assert abs(e_at(x=x)) < 1


@pytest.mark.parametrize("psi", np.linspace(-6.5, 6.5, 14))
def test_dead_zone_heading(psi):
    assert abs(e_at(psi_deg=psi)) < 1


@pytest.mark.parametrize("x, psi", [(18, 0), (-18, 0), (0, 10), (0, -10)])
def test_outside_dead_zone(x, psi):
    assert abs(e_at(x, psi)) > 100


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


def test_unswapped_wiring_steers_away():
    """Белое выше + моторы не переставлены: код урока рулит от линии (поэтому в подключении есть перестановка)."""
    traj = model.simulate(LINE, (0.0, 0.0, np.radians(12)), t_end=0.1, motor=Motor(swapped=False))
    assert traj.psi[-1] > traj.psi[0]


def test_cache_key_sees_defaults(tmp_path, monkeypatch):
    """Ключ кэша строится по всем параметрам: явный Motor() и умолчание дают один файл, другой Motor — второй."""
    monkeypatch.setattr(model, "CACHE", tmp_path)
    model.cached_simulate(LINE, (0.0, 0.0, 0.2), t_end=0.05)
    model.cached_simulate(LINE, (0.0, 0.0, 0.2), t_end=0.05, motor=Motor())
    assert len(list(tmp_path.glob("*.npz"))) == 1
    model.cached_simulate(LINE, (0.0, 0.0, 0.2), t_end=0.05, motor=Motor(swapped=False))
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


@pytest.mark.parametrize("s_swap, m_swap, toward", [(False, True, True), (True, False, True),
                                                   (False, False, False), (True, True, False)])
def test_wiring_combinations(s_swap, m_swap, toward):
    """Белое выше: робот рулит к линии, только если переставлено ровно одно — датчики или моторы."""
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
