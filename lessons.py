"""Картинки и видео сцен для репозитория уроков — одной командой (кнопка ▶ или `python lessons.py`).

Для каждой сцены: видео 1080p (рендер; без изменений Manim берёт его из кэша) → ключевой кадр
(render(..., key=True)) → out/lessons/<имя>.png 1280×720 и out/lessons/<имя>.mp4 (720p, без звука, H.264 CRF 26).
Ключевой кадр задаёт сама сцена: key_time — момент видео, key = True — сцена рисует кадр сама (сцена 3).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scenes.base import render, video_path  # noqa: E402
from scenes.s0_motion import Scene0  # noqa: E402
from scenes.s1_error import Scene1  # noqa: E402
from scenes.s2a_p import Scene2a  # noqa: E402
from scenes.s2b_turn import Scene2b  # noqa: E402
from scenes.s2c_d import Scene2c  # noqa: E402
from scenes.s3_cross import Scene3  # noqa: E402

OUT = ROOT / "out" / "lessons"
SCENES = [(Scene0, "scene-0-motion"), (Scene1, "scene-1-error"), (Scene2a, "scene-2a-p"),
          (Scene2b, "scene-2b-turn"), (Scene2c, "scene-2c-d"), (Scene3, "scene-3-cross")]
LIMITS = {".png": 300_000, ".mp4": 4_000_000}      # цель по размеру: это git, каждая версия хранится навсегда


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def build(scene_cls, name, quality="1080p"):
    render(scene_cls, quality, preview=False)
    key = render(scene_cls, quality, key=True)
    png, mp4 = OUT / f"{name}.png", OUT / f"{name}.mp4"
    ffmpeg("-i", key, "-vf", "scale=1280:720:flags=lanczos", "-compression_level", 100, png)
    ffmpeg("-i", video_path(scene_cls, quality), "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", 26,
           "-preset", "slow", "-an", "-movflags", "+faststart", mp4)
    return png, mp4


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene_cls, name in SCENES:
        for f in build(scene_cls, name):
            size = f.stat().st_size
            flag = "  ← больше цели" if size > LIMITS[f.suffix] else ""
            rows.append(f"{f.name:22} {size / 1e3:8.0f} КБ{flag}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
