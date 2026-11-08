"""Камера: точки поля (мм) → экран. Север всегда вверх.

follow — середина оси робота всегда в точке anchor, поле едет под роботом;
fixed  — камера неподвижна: точка поля center стоит в anchor, робот ездит по экрану.
"""
import numpy as np


class Camera:
    def __init__(self, fr, mode="follow", center=(0.0, 0.0), anchor=None):
        self.fr, self.mode = fr, mode
        self.center = np.asarray(center, float)
        self.anchor = fr.anchor if anchor is None else np.asarray(anchor, float)

    def follow(self, x, y):
        """Сообщить камере, где робот; в режиме follow она за ним сдвигается."""
        if self.mode == "follow":
            self.center = np.array([x, y], float)

    def to_screen(self, pts_mm):
        p = (np.asarray(pts_mm, float).reshape(-1, 2) - self.center) * self.fr.mm
        return np.column_stack([p, np.zeros(len(p))]) + self.anchor

    def point(self, x, y):
        return self.to_screen([[x, y]])[0]
