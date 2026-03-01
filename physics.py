from __future__ import annotations

from objects import Wireframe3DObject

LINEAR_DRAG = 0.98
ANGULAR_DRAG = 0.995
WALL_BOUNCE_RESTITUTION = 0.6
IDLE_ANGULAR_SPEED = 0.25  # rad/s gentle Y-axis rotation
IDLE_THRESHOLD = 0.05  # rad/s


class PhysicsEngine:
    def __init__(self, screen_width: int, screen_height: int):
        self.w = screen_width
        self.h = screen_height

    def step(self, objects: list[Wireframe3DObject], dt: float) -> None:
        for obj in objects:
            if obj.grabbed:
                continue

            # --- Linear integration ---
            obj.x += obj.vx * dt
            obj.y += obj.vy * dt

            # Linear drag
            obj.vx *= LINEAR_DRAG
            obj.vy *= LINEAR_DRAG

            # --- Angular integration ---
            obj.rot_x += obj.angular_vx * dt
            obj.rot_y += obj.angular_vy * dt
            obj.rot_z += obj.angular_vz * dt

            # Angular drag
            obj.angular_vx *= ANGULAR_DRAG
            obj.angular_vy *= ANGULAR_DRAG
            obj.angular_vz *= ANGULAR_DRAG

            # --- Wall bounce ---
            margin = 30.0
            if obj.x < margin:
                obj.x = margin
                obj.vx = abs(obj.vx) * WALL_BOUNCE_RESTITUTION
            elif obj.x > self.w - margin:
                obj.x = self.w - margin
                obj.vx = -abs(obj.vx) * WALL_BOUNCE_RESTITUTION

            if obj.y < margin:
                obj.y = margin
                obj.vy = abs(obj.vy) * WALL_BOUNCE_RESTITUTION
            elif obj.y > self.h - margin:
                obj.y = self.h - margin
                obj.vy = -abs(obj.vy) * WALL_BOUNCE_RESTITUTION

            # --- Idle rotation ---
            total_angular = (
                obj.angular_vx ** 2 + obj.angular_vy ** 2 + obj.angular_vz ** 2
            ) ** 0.5
            if total_angular < IDLE_THRESHOLD:
                obj.angular_vy = IDLE_ANGULAR_SPEED
