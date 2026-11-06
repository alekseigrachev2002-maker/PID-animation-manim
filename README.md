# ПИД на роботе по линии — анимации Manim

Решения и сюжет сцен: `docs/spec.md`. Замеры, которые ждёт модель: `docs/measurements.md`. Исходники робота: `assets/`.

```
pip install numpy scipy matplotlib pytest
python -m pytest -q          # знак руления, e = 0 на центре, мёртвая зона датчиков
python quicklook.py all      # картинки для подбора параметров → out/
```

- `params.py` — параметры (мм, с, ШИМ, АЦП); заглушки помечены.
- `track.py` — осевые линии трасс и расстояние до них.
- `model.py` — датчики, моторы, кинематика, регулятор; `simulate()`, `prescribed()`, кэш `.cache/`.
- `quicklook.py` — графики matplotlib.
