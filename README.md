# ПИД на роботе по линии — анимации Manim

- `docs/spec.md` — решения и сюжет сцен; `docs/architecture.md` — как устроен код; `docs/measurements.md` — замеры.
- Запуск сцены — кнопкой ▶ в файле `scenes/…py`: `render(Scene1, "4k")`, качество `"240p"`, `"480p"`, `"720p"`, `"1080p"`, `"4k"`. Все видео — в папке `videos/` (`Scene1_4k.mp4`), кадры (`still=True`) — в `renders/`.
- Тесты физики — `python -m pytest -q`.
