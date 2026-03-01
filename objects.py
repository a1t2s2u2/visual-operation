from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class DraggableObject(Protocol):
    x: float
    y: float
    color: tuple[int, int, int]
    grabbed: bool
    hover: bool

    def contains(self, point: np.ndarray) -> bool: ...


@dataclass
class DraggableRect:
    x: float
    y: float
    width: float
    height: float
    color: tuple[int, int, int]
    grabbed: bool = False
    hover: bool = False

    def contains(self, point: np.ndarray) -> bool:
        px, py = float(point[0]), float(point[1])
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x - half_w <= px <= self.x + half_w
            and self.y - half_h <= py <= self.y + half_h
        )


@dataclass
class DraggableCircle:
    x: float
    y: float
    radius: float
    color: tuple[int, int, int]
    grabbed: bool = False
    hover: bool = False

    def contains(self, point: np.ndarray) -> bool:
        px, py = float(point[0]), float(point[1])
        dx = px - self.x
        dy = py - self.y
        return dx * dx + dy * dy <= self.radius * self.radius


def create_default_objects() -> list[DraggableRect | DraggableCircle]:
    return [
        DraggableRect(x=120, y=150, width=120, height=80, color=(200, 100, 50)),
        DraggableRect(x=500, y=120, width=90, height=90, color=(50, 180, 50)),
        DraggableCircle(x=320, y=300, radius=50, color=(50, 50, 200)),
        DraggableCircle(x=200, y=380, radius=35, color=(30, 200, 200)),
        DraggableRect(x=450, y=350, width=110, height=70, color=(180, 50, 180)),
    ]
