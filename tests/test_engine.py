"""Возможности движка, пришедшие из нового курса: ПИД с И, счет перекрестков по фронту, ускорение проигрывания,
дальномер, поворот камеры."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Sonar  # noqa: E402
from scenes.base import PLAYBACK, warp  # noqa: E402

LINE = track.Route(start=(0, -300), heading="вверх").forward(3300)
CROSSES = [track.Route(start=(-300, y), heading="вправо").forward(600) for y in (300, 600, 900)]
FIELD = track.Track(LINE, *CROSSES)
START = (0.0, 0.0, 0.0)


def test_pid_ki_zero_is_pd():
    """ki = 0: uI нулевой, ШИМ — как у ПД: speed ± (uP + uD)."""
    c = Controller(kp=0.2, kd=1.0)
    run = model.simulate(FIELD, (0.0, 0.0, np.radians(10)), ctrl=c, t_end=1.0)
    assert np.all(run.uI == 0)
    pwm = np.clip(c.speed + run.uP + run.uD, -255, 255)
    assert np.allclose(run.pwmL, pwm)


def test_integral_is_sum_of_err():
    run = model.simulate(FIELD, (0.0, 0.0, np.radians(10)), ctrl=Controller(kp=0.2, kd=1.0), ki=0.004, t_end=0.5)
    assert np.allclose(run.integral, np.cumsum(run.e))
    assert np.allclose(run.uI, 0.004 * run.integral)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_stop_at_nth_crossing(n):
    """Стоп на n-м перекрестке: прошлые робот проезжает, счет по фронту."""
    run = model.simulate(FIELD, START, ctrl=Controller(kp=0.2, kd=1.0), t_end=8.0, stop_below=300, stop_at=n)
    sensors_y = run.y[-1] + 94.714
    assert sensors_y == pytest.approx([300, 600, 900][n - 1], abs=15)


def test_start_crossing_not_counted():
    """Робот стартует датчиками на перекрестке — он не в счет, стоп на следующем."""
    start = (0.0, 300 - 94.714, 0.0)
    run = model.simulate(FIELD, start, ctrl=Controller(kp=0.2, kd=1.0), t_end=8.0, stop_below=300, stop_at=1)
    assert run.y[-1] + 94.714 == pytest.approx(600, abs=15)


def test_warp_monotonic_and_bounded():
    total, rate, tau, k = warp(0.0, 6.0, [(0.0, 1.0), (5.0, 6.0)], fast=3.0)
    assert np.all(np.diff([rate(a) for a in np.linspace(0, 1, 200)]) >= 0)
    assert rate(0.0) == pytest.approx(0.0) and rate(1.0) == pytest.approx(1.0, abs=1e-3)
    assert k.min() >= 1.0 and k.max() <= 3.0 + 1e-9
    assert np.all(k[(tau > 0.0) & (tau < 1.0)] < 1.01)           # в зоне — без ускорения
    assert total < 6.0 / PLAYBACK                                   # вне зон — быстрее


def test_warp_without_fast_is_plain():
    total, rate, _, _ = warp(0.0, 2.0, [], fast=1.0)
    assert total == pytest.approx(2.0 / PLAYBACK, rel=1e-3)
    assert rate(0.5) == pytest.approx(0.5, abs=1e-3)


def test_sonar_sees_block_ahead_only():
    s = Sonar()
    block = track.Block(center=(0.0, 300.0), side=50.0)
    d = model.sonar_distance((0.0, 0.0, 0.0), [block], s)
    assert d == pytest.approx(300 - 25 - s.front)
    assert model.sonar_distance((0.0, 0.0, np.radians(90)), [block], s) == 600.0      # смотрит налево — не видит
    assert model.sonar_distance((0.0, 0.0, 0.0), [], s) == 600.0


def test_camera_rotation():
    from visuals.camera import Camera
    from visuals.layout import Frame
    fr = Frame()
    plain, rot0 = Camera(fr, "fixed"), Camera(fr, "fixed", rot=0.0)
    pts = [[10.0, 20.0], [-5.0, 7.0]]
    assert np.array_equal(plain.to_screen(pts), rot0.to_screen(pts))
    turned = Camera(fr, "fixed", rot=np.pi / 2)                    # +y (вперед) → влево на экране
    p = turned.to_screen([[0.0, 100.0]])[0] - turned.anchor
    assert p[0] < 0 and abs(p[1]) < 1e-9
