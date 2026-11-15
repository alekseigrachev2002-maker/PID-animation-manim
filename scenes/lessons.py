"""Видео и картинки сцен — кнопка ▶ в этом файле.

Видео 1080p всех шести сцен (0, 1, 2a, 2b, 2c, 3) — в videos/: <Сцена>_1080p.mp4, для презентации (на весь слайд).
Картинки (PNG 1280×720) — в renders/lessons/, по одной на сцену: кадр, по которому без видео понятно, что происходит;
они же — кадры-превью видео в презентации (Видео → Кадр-постер → Изображение из файла).
Момент кадра записан здесь, в KEYS (секунды видео сцены); кадр вырезается из видео 1080p.
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
from manim import ValueTracker  # noqa: E402
from PIL import Image  # noqa: E402

import model  # noqa: E402
from params import Controller  # noqa: E402
from scenes.base import render  # noqa: E402
from scenes.s0_motion import Scene0  # noqa: E402
from scenes.s1_error import Scene1  # noqa: E402
from scenes.s2a_p import Scene2a  # noqa: E402
from scenes.s2b_turn import Scene2b  # noqa: E402
from scenes.s2c_d import Scene2c  # noqa: E402
from scenes.s3_cross import Scene3  # noqa: E402
from track import Track, field1  # noqa: E402
from visuals.panel import GraphPanel, Signal  # noqa: E402

OUT = ROOT / "renders" / "lessons"
QUALITY = "1080p"                     # видео для презентации; из него же — картинки
SIZE = (1280, 720)
LIMIT = 300_000


class Scene3Key(Scene3):
    """Сцена 3 для картинки: тот же сюжет, но график — только езда и стоп, своя шкала времени от начала езды."""

    def construct(self):
        st, sens = self.st, self.sensor
        track = Track(*field1().lines)
        segs = self.segments(track)
        ends = np.cumsum([sg.t[-1] for sg in segs])
        traj = model.chain(*segs)
        self.show(track, traj)
        graph, t0 = model.chain(segs[3], segs[4]), ends[2]
        a0, a1 = model.channel_sides(sens)
        L, R = Signal(f"L — {a0} датчик, A0", "L", "L"), Signal(f"R — {a1} датчик, A1", "R", "R")
        panel = GraphPanel(self.fr, st, "Перекресток", "оба ниже порога: стоп", y=("АЦП", 0, 1023),
                           t_max=graph.t[-1] * 1.08, t_ticks=np.arange(0, graph.t[-1], 0.5), signals=[L, R])
        trs = Controller().threshold
        limit = panel.plot.hline(trs, f"threshold = {trs:g}", ValueTracker(1.0))
        self.add(panel, limit, *panel.curves(graph, lambda: max(self.t.get_value() - t0, 0.0)))
        self.play(panel.set_formula("L < threshold && R < threshold", [("L", "L"), ("R", "R")]), run_time=0.5)
        self.advance(ends[4], speed=1.0)                  # в реальном времени: нужен только последний кадр


# картинки: (сцена, момент видео в секундах или None — последний кадр, имя файла)
KEYS = [
    (Scene0, 23.5, "scene-0-motion"),       # третий маневр motors(180, 150): робота увело вправо, ШИМ — разница
    (Scene1, 11.8, "scene-1-error"),        # робот чуть повернут, L и R бледные, err на плато ≈ −700
    (Scene2a, 17.6, "scene-2a-p"),          # хороший P (kp = 0.15): конец прогона, робот сам вышел на линию и встал
    (Scene2b, 26.0, "scene-2b-turn"),       # kp = 0.12: съехал с дуги и остановился
    (Scene2c, 25.0, "scene-2c-d"),          # kd = 1: вышел на прямую; uD — всплеск на входе в дугу и на выходе
    (Scene3Key, None, "scene-3-cross"),     # стоп на перекрестке: L и R оба ниже порога
]


def video(scene_cls):
    """Видео сцены 1080p — из него вырезается кадр (рендер; без изменений Manim берет части из кэша)."""
    render(scene_cls, QUALITY, preview=False)
    return ROOT / "videos" / f"{scene_cls.__name__}_{QUALITY}.mp4"


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for scene_cls, t, name in KEYS:
        path = video(scene_cls)
        save_png(frame_at(path, t), OUT / f"{name}.png")
        files.append(OUT / f"{name}.png")
        if scene_cls is Scene3Key:
            path.unlink()                                 # служебное видео — только ради картинки
    video(Scene3)                                         # обычное видео сцены 3 — для презентации
    for f in files:
        size = f.stat().st_size
        print(f"{f.name:22} {size / 1e3:8.0f} КБ" + ("  ← больше цели" if size > LIMIT else ""))


if __name__ == "__main__":
    main()
