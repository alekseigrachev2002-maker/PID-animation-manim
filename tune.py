"""Настройка ПД по модели — тем же алгоритмом, что в уроке 2, на повороте сцены 2b (R = 150 мм, поле 2).

1. Поднимаем kp, пока робот не впишется в поворот.
2. Опускаем kp, пока не перестанет вписываться — «почти».
3. С этим kp поднимаем kd, пока снова не впишется.

Печатает каждый шаг; по этим числам выбраны kp и kd сцен 2a–3. Запуск: python tune.py (несколько минут).
"""
import numpy as np

import model
from params import Controller, Motor
from scenes.s2b_turn import Scene2b

DELAY = Scene2b.delay              # задержка команда → колесо, как в сценах
T_RUN = Scene2b.t_run


def fits(track, pose, kp, kd=0.0):
    """Вписался: за прогон не ушел от линии, и в конце она между датчиками (а не сбоку от робота)."""
    run = model.simulate(track, pose, ctrl=Controller(kp=kp, kd=kd), t_end=T_RUN, motor=Motor(delay=DELAY))
    return bool(np.isnan(run.lost_at)) and not model.off_line(track, (run.x[-1], run.y[-1], run.psi[-1]))


def main():
    scene = Scene2b()
    track = scene.track = scene.make_track()
    pose = tuple(float(v) for v in scene.start(0))

    kp = 0.01
    while not fits(track, pose, kp):
        print(f"kp = {kp:.2f}: вылет")
        kp = round(kp + 0.01, 2)
    print(f"kp = {kp:.2f}: вписался")

    almost = round(kp - 0.01, 2)                       # шаг вниз — уже не вписывается
    print(f"kp = {almost:.2f}: почти — вылет")

    kd = 0.0
    while not fits(track, pose, almost, kd):
        print(f"kp = {almost:.2f}, kd = {kd:.1f}: вылет")
        kd = round(kd + 0.1, 1)
    print(f"kp = {almost:.2f}, kd = {kd:.1f}: вписался")
    print(f"\nИтог: kp = {almost:.2f}, kd от {kd:.1f} (в сценах — с запасом)")


if __name__ == "__main__":
    main()
