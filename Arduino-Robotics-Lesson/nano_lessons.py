"""Видео и картинки сцен нового курса — кнопка ▶ в этом файле или python Arduino-Robotics-Lesson/nano_lessons.py.

Видео всех сцен рендерятся в videos/ (<Сцена>_<качество>.mp4, общая папка с олимпиадными сценами) и копируются
с именами курса в out/arduino/: scene-<урок><буква>-<тема>.mp4 и .png (кадр из того же видео, 1920×1080).
Момент кадра — в KEYS, через промежутки BLOCK. Черновик: python Arduino-Robotics-Lesson/nano_lessons.py 480p
"""
from nano_base import BLOCK, ROOT, render  # первым: добавляет в путь корень репозитория

import shutil
import sys

from n1a_motion import Nano1a
from n2a_error import Nano2a
from n2b_p import Nano2b
from n2c_d import Nano2c
from n3a_i_offset import Nano3a
from n3b_cross import Nano3b
from scenes.lessons import END, LIMIT, frame_at, save_png

OUT = ROOT / "out" / "arduino"
QUALITY = "1080p"

KEYS = [
    (Nano1a, 3 * BLOCK - END, "scene-1a-motion"),       # дуга вправо: левое колесо быстрее
    (Nano2a, BLOCK + 4.5, "scene-2a-error"),            # второй проход: err на плато ≈ −700
    (Nano2b, 3 * BLOCK - END, "scene-2b-p"),            # kp = 0,3: в повороте
    (Nano2c, 2 * BLOCK - END, "scene-2c-d"),            # kd = 1: вписался
    (Nano3a, 2 * BLOCK - END, "scene-3a-i-offset"),     # с И: uI взял поворот на себя
    (Nano3b, BLOCK - END, "scene-3b-cross"),            # перекресток X позади
]


def main(quality=QUALITY):
    OUT.mkdir(parents=True, exist_ok=True)
    for scene_cls, t, name in KEYS:
        render(scene_cls, quality, preview=False)
        video = ROOT / "videos" / f"{scene_cls.__name__}_{quality}.mp4"
        shutil.copyfile(video, OUT / f"{name}.mp4")
        png = OUT / f"{name}.png"
        save_png(frame_at(video, t), png)
        size = png.stat().st_size
        print(f"{name:22} {size / 1e3:8.0f} КБ" + ("  ← больше цели" if size > LIMIT else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else QUALITY)
