"""Правила панели и сцен, которые легко сломать незаметно (без рендера видео)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model  # noqa: E402
from params import Geometry, Style  # noqa: E402
from track import field1  # noqa: E402
from visuals.layout import Frame  # noqa: E402
from visuals.panel import ERR, GraphPanel, Signal  # noqa: E402

FR, ST = Frame(), Style()
L, R = Signal("L", "L", "L"), Signal("R", "R", "R")
TT = np.linspace(0, 1, 50)
TRAJ = SimpleNamespace(tt=TT, L=np.full(50, 939.0), R=np.full(50, 939.0), e=np.zeros(50))   # совпадающие кривые


def panel(signals, **kw):
    return GraphPanel(FR, ST, "Заголовок", "пояснение", ("АЦП", 0, 1023), 1.2, [0, 1], signals, **kw)


def y_px(curve):
    """Средняя высота ломаной кривой, px эталонного кадра."""
    return float(np.mean(curve[0].points[:, 1])) / FR.unit


def test_two_curves_constant_offset_half_width():
    """Две кривые: первая на +x, вторая на −x, x — полтолщины линии; одинаково по всей сцене."""
    p = panel([L, R])
    cL, cR = p.curves(TRAJ, lambda: 1.0)
    base = y_px(p.curve(TRAJ, L, lambda: 1.0))
    assert y_px(cL) - base == pytest.approx(L.width_px / 2, abs=1e-6)
    assert y_px(cR) - base == pytest.approx(-R.width_px / 2, abs=1e-6)


def test_single_curve_and_separate_curve_not_shifted():
    p = panel([ERR])
    (c,) = p.curves(TRAJ, lambda: 1.0)
    assert y_px(c) == pytest.approx(y_px(p.curve(TRAJ, ERR, lambda: 1.0)), abs=1e-9)   # err в сцене 1 — так же


def test_header_text_in_place_from_first_frame():
    """Формула и пояснение стоят сразу; пояснение — вплотную за блоком формулы."""
    h = panel([L], formula="err = L - R", formula_colors=[("err", "e")]).header
    assert h.block is not None and h.block in h.submobjects
    assert h.caption.get_left()[0] == pytest.approx(h.block.get_right()[0] + 16 * FR.unit)


def test_set_text_swaps_caption_to_new_place():
    """Формула сменила ширину — пояснение встает за новым блоком (swap), а не съезжает вбок."""
    h = panel([L], formula="x").header
    h.set_text(formula="control = err * kp + (err - errOld) * kd")
    assert h.caption.get_left()[0] == pytest.approx(h.block.get_right()[0] + 16 * FR.unit)
    assert h.caption_text == "пояснение"


def test_scene0_half_turn_is_180():
    from scenes.s0_motion import Scene0
    track, pose = field1(), (600.0, 600.0, 0.0)
    t_on = Scene0.half_turn(SimpleNamespace(geom=Geometry()), track, pose, (-150, 150))
    psi = model.drive(track, pose, (-150, 150), t_on, stop=True).psi[-1]
    assert np.degrees(abs(psi)) == pytest.approx(180, abs=1.0)


def test_run_series_scalar_is_p_only():
    """Прогон, заданный одним числом, — чистый P: kd не берется из значения по умолчанию в params."""
    from scenes.s2b_turn import Scene2b
    s = Scene2b()
    s.track = s.make_track()
    run = s.simulate(tuple(float(v) for v in s.start(0)), 0.09)
    assert np.all(run.uD == 0)
