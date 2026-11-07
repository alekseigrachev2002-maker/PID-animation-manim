import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Geometry, Motor, Sensor  # noqa: E402

G, S, C = Geometry(), Sensor(), Controller()
LINE = track.straight()


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
