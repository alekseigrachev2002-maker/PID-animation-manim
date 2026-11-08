# Как устроен проект

Сначала физика считает траекторию, потом сцена её проигрывает.

```
params.py   все числа: робот, датчики, моторы, регулятор, цвета, макет кадра
track.py    трасса: Track(Route(...).forward(...).right(R=...)), метки, поза на линии
model.py    физика: датчики → err → регулятор → ШИМ → моторы → движение; simulate() → Trajectory
            (таблица по времени: поза, путь колёс, L, R, err, uP, uD, ШИМ); prescribed()/move() — движение «руками»

visuals/layout.py  перевод пикселей и мм в единицы Manim, текст, фон
visuals/camera.py  камера: follow (робот в одной точке экрана, поле едет) или fixed (поле стоит, робот ездит)
visuals/field.py   линии трассы на экране
visuals/robot.py   робот: place(точка, курс), протектор, метки датчиков; Robot(sensors=False) — без датчиков
visuals/panel.py   панель графиков: шапка, блок формулы, график сигналов, легенда, доп. легенда (ШИМ колёс)

scenes/base.py     PIDScene — стенд: время t, show(трасса, траектория, камера, элементы), advance, return_to
                   RunSeries — серия прогонов: прогон → стоп → возврат → следующий
scenes/s*.py       сцены
```

Каждый кадр стенд берёт состояние траектории в момент t и раздаёт его камере, полю, роботу и элементам
(например строке ШИМ). Кривые на графике сами дорисовываются до t.

## Новая сцена с прогонами регулятора
```python
class Scene2c(RunSeries):
    phrase, formula, law = "Вводим D", "control = kp·err + kd·Δerr", "control"
    signals = [Signal("uP", "uP", "e"), Signal("uD", "uD", "D")]
    runs = [(dict(kp=0.3, kd=0), "без D"), (dict(kp=0.3, kd=2), "с D")]

    def make_track(self):
        return Track(Route(start=(0, -1500), heading="вверх").forward(1500).mark("поворот").right(R=100).forward(1500))

    def start(self, i):
        return self.track.pose("поворот", -350)
```
Сцена без регулятора (как сцена 1) — `PIDScene`: своя траектория, `self.show(...)`, `GraphPanel(...)`, `self.advance(...)`.

## Проверка
`python -m pytest -q` — физика. Кадр сцены — `render(Scene, "high_quality", still=True)`.
