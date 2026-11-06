"""Быстрые графики matplotlib для подбора параметров. Запуск: python quicklook.py [s1|s2a|s2b|all]."""
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import model  # noqa: E402
import track  # noqa: E402
from params import Controller, Motor  # noqa: E402

OUT = Path(__file__).parent / "out"
COL = {"L": "#4C8DF6", "R": "#F59E0B", "e": "#34D399", "D": "#E879F9"}


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def scene1_motion(amp_deg=18.0, period=2.0, cycles=2, dt=0.005):
    t = np.arange(0, cycles * period, dt)
    return t, np.radians(amp_deg) * np.sin(2 * np.pi * t / period)


def plot_s1():
    t, psi = scene1_motion()
    tr = model.prescribed(track.straight(), t, 0.0, 0.0, psi)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(tr.tt, tr.L, color=COL["L"], label="L")
    a1.plot(tr.tt, tr.R, color=COL["R"], label="R")
    a1.plot(tr.tt, tr.e, color=COL["e"], label="e = L − R")
    a1.set_ylabel("АЦП"); a1.legend(loc="upper right"); a1.grid(alpha=.3)
    a2.plot(t, np.degrees(psi), color="k"); a2.set_ylabel("ψ, °"); a2.set_xlabel("t, с"); a2.grid(alpha=.3)
    save(fig, "s1")


def plot_runs(trajs, labels, name, title, track_obj=None, xlim=None):
    """Курс, ошибка, u-части и путь для серии прогонов."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    ax_e, ax_u, ax_p = (fig.add_subplot(gs[i, 0]) for i in range(3))
    ax_xy = fig.add_subplot(gs[:, 1])
    for tj, lab in zip(trajs, labels):
        line, = ax_e.plot(tj.tt, tj.e, label=lab)
        c = line.get_color()
        ax_u.plot(tj.tt, tj.uP, color=c)
        ax_u.plot(tj.tt, tj.uD, color=c, ls=":")
        ax_p.plot(tj.tt, tj.pwmL, color=c)
        ax_p.plot(tj.tt, tj.pwmR, color=c, ls="--")
        ax_xy.plot(tj.x, tj.y, color=c)
        if not np.isnan(tj.lost_at):
            ax_xy.plot(tj.x[-1], tj.y[-1], "x", color=c, ms=10)
    for ln in (track_obj.lines if track_obj else []):
        ax_xy.plot(ln.points[:, 0], ln.points[:, 1], color="#151515", lw=3, alpha=.25)
    for p in (Motor().pwm_stop, Motor().pwm_start):
        ax_p.axhline(p, color="gray", lw=.8, ls="-.")
    ax_e.set_ylabel("e"); ax_u.set_ylabel("uP (—), uD (···)"); ax_p.set_ylabel("ШИМ L (—), R (--)")
    ax_p.set_xlabel("t, с"); ax_e.legend(fontsize=8)
    if xlim:
        ys = np.concatenate([tj.y for tj in trajs])
        ax_xy.set_xlim(*xlim); ax_xy.set_ylim(ys.min() - 50, ys.max() + 50)
    else:
        ax_xy.set_aspect("equal")
    ax_xy.set_title("путь середины оси, мм")
    for a in (ax_e, ax_u, ax_p, ax_xy):
        a.grid(alpha=.3)
    fig.suptitle(title)
    save(fig, name)


def plot_s2a(kps=(0.1, 0.3, 0.5, 0.8), psi0=15.0, t_end=2.5, delay=0.02):
    tr = track.straight(y1=6000)
    motor = Motor(delay=delay)
    trajs = [model.cached_simulate(tr, (0.0, 0.0, np.radians(psi0)), ctrl=Controller(kp=k), motor=motor,
                                   t_end=t_end) for k in kps]
    plot_runs(trajs, [f"kp={k}" for k in kps], "s2a",
              f"Подбор P: старт с поворотом {psi0}°, задержка {delay * 1000:.0f} мс", tr, xlim=(-60, 60))


def plot_s2b(kp=0.08, kds=(0.0, 2.0), size=(1200, 900), radii=(220, 95, 240, 95), t_end=16.0):
    tr = track.rounded_rect(*size, radii)
    pose0 = (size[0] / 2, 0.0, 0.0)
    trajs = [model.cached_simulate(tr, pose0, ctrl=Controller(kp=kp, kd=kd), t_end=t_end) for kd in kds]
    plot_runs(trajs, [f"kd={kd}" for kd in kds], "s2b", f"Вводим D: kp={kp}, радиусы {radii}", tr)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for key, fn in (("s1", plot_s1), ("s2a", plot_s2a), ("s2b", plot_s2b)):
        if what in (key, "all"):
            fn()
