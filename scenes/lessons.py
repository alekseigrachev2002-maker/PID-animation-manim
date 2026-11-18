"""Видео и картинки сцен — кнопка ▶ в этом файле.

Видео 1080p всех шести сцен (0, 1, 2a, 2b, 2c, 3) — в videos/: <Сцена>_1080p.mp4, для презентации (на весь слайд).
Картинки (PNG 1920×1080) — в renders/lessons/, по одной на сцену: кадр, по которому без видео понятно, что происходит;
они же — кадры-превью видео в презентации (Видео → Кадр-постер → Изображение из файла).
Момент кадра записан здесь, в KEYS (через промежутки BLOCK); кадр вырезается из видео.
Сцена 3 — особый кадр: на графике только езда по линии и стоп, без провала правого датчика при развороте
на линию (без видео он непонятен). Его рисует Scene3Key — та же сцена, другой график; ее видео служебное и после
вырезки кадра удаляется, в videos/ остается обычное видео сцены 3.

Цель по размеру (это git): PNG до ~300 КБ. Без внешнего ffmpeg: кадры — через PyAV (стоит вместе с Manim).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import model  # noqa: E402
from scenes.base import BLOCK, render  # noqa: E402
from scenes.s0_motion import Scene0  # noqa: E402
from scenes.s1_error import Scene1  # noqa: E402
from scenes.s2a_p import Scene2a  # noqa: E402
from scenes.s2b_turn import Scene2b  # noqa: E402
from scenes.s2c_d import Scene2c  # noqa: E402
from scenes.s3_cross import Scene3  # noqa: E402
from track import Track, field1  # noqa: E402

OUT = ROOT / "renders" / "lessons"
QUALITY = "1080p"                     # видео для презентации; из него же — картинки. Черновик: python scenes/lessons.py 480p
SIZE = (1920, 1080)                   # как видео 1080p — превью в презентации и картинки уроков
LIMIT = 300_000


class Scene3Key(Scene3):
    """Сцена 3 для картинки: тот же сюжет, но график — только езда и стоп, своя шкала времени от начала езды."""

    def construct(self):
        track = Track(*field1().lines)
        segs = self.segments(track)
        ends = np.cumsum([sg.t[-1] for sg in segs])
        traj = model.chain(*segs)
        self.show(track, traj)
        graph, t0 = model.chain(segs[3], segs[4]), ends[2]
        panel, limit = self.panel("оба ниже порога: стоп", graph.t[-1], np.arange(0, graph.t[-1], 0.5))
        self.add(panel, limit, *panel.curves(graph, lambda: max(self.t.get_value() - t0, 0.0)))
        self.wait(0.1, frozen_frame=False)
        self.advance(ends[4], speed=1.0)                  # в реальном времени: нужен только последний кадр


# картинки: (сцена, момент видео в секундах или None — последний кадр, имя файла)
# Сцены состоят из промежутков BLOCK: момент — «конец промежутка k минус пауза», поэтому не едет от правок внутри.
END = 0.5   # за столько секунд до конца промежутка — все дорисовано, робот стоит
KEYS = [
    (Scene0, 3 * BLOCK - END, "scene-0-motion"),    # третий маневр motors(100, 75): робота увело вправо, ШИМ — разница
    (Scene1, BLOCK + 4.5, "scene-1-error"),         # второй проход: робот повернут, L и R бледные, err на плато ≈ −700
    (Scene2a, 2 * BLOCK - END, "scene-2a-p"),       # хороший P (kp = 0.15): конец прогона, робот сам вышел на линию
    (Scene2b, 2 * BLOCK - END, "scene-2b-turn"),    # kp = 0.1: съехал с дуги и остановился
    (Scene2c, 2 * BLOCK - END, "scene-2c-d"),       # kd = 1: вышел на прямую; uD — всплеск на входе в дугу и на выходе
    (Scene3Key, None, "scene-3-cross"),             # стоп на перекрестке: L и R оба ниже порога
]


def video(scene_cls, quality=QUALITY):
    """Видео сцены — из него вырезается кадр (рендер; без изменений Manim берет части из кэша)."""
    render(scene_cls, quality, preview=False)
    return ROOT / "videos" / f"{scene_cls.__name__}_{quality}.mp4"


def frame_at(path, t=None):
    """Кадр видео на момент t, с (первый кадр не раньше t), или последний кадр (t = None) — как картинка PIL."""
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        if t is not None:
            box.seek(int(max(t - 1.0, 0) / stream.time_base), stream=stream)    # к ключевому кадру до t
        last = None
        for fr in box.decode(stream):
            last = fr
            if t is not None and fr.time >= t - 1e-6:
                break
        return last.to_image()


def save_png(img, path):
    img.convert("RGB").resize(SIZE, Image.LANCZOS).save(path, optimize=True)


def main(quality=QUALITY):
    """Видео всех сцен в качестве quality и картинки из них. Моменты KEYS — те же в любом качестве."""
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for scene_cls, t, name in KEYS:
        path = video(scene_cls, quality)
        save_png(frame_at(path, t), OUT / f"{name}.png")
        files.append(OUT / f"{name}.png")
        if scene_cls is Scene3Key:
            path.unlink()                                 # служебное видео — только ради картинки
    video(Scene3, quality)                                # обычное видео сцены 3 — для презентации
    for f in files:
        size = f.stat().st_size
        print(f"{f.name:22} {size / 1e3:8.0f} КБ" + ("  ← больше цели" if size > LIMIT else ""))


if __name__ == "__main__":
    # Кнопка ▶ или python scenes/lessons.py — 1080p для презентации; python scenes/lessons.py 480p — быстрый черновик
    main(sys.argv[1] if len(sys.argv) > 1 else QUALITY)
