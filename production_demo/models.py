"""Small data types shared by the production-line animation modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class JarSpec:
    index: int
    body: str
    mouth_site: str
    neck_site: str
    top_leaf: str
    bottom_leaf: str
    top_pick_site: str
    bottom_pick_site: str
    top_suction_weld: str
    bottom_suction_weld: str
    top_table_weld: str
    bottom_table_weld: str
    top_mouth_weld: str
    bottom_mouth_weld: str


@dataclass
class PathEvent:
    point_index: int
    label: str
    callback: object


@dataclass
class JointPath:
    """Continuous joint-space playback with callbacks at path endpoints."""

    name: str
    joint_addrs: np.ndarray
    dof_addrs: np.ndarray
    points: list[np.ndarray]
    durations: list[float]
    events: list[PathEvent] = field(default_factory=list)
    elapsed: float = 0.0
    fired: set[int] = field(default_factory=set)
    complete: bool = False

    @property
    def duration(self) -> float:
        return float(sum(self.durations))

    def advance(self, data, dt: float):
        if self.complete:
            return
        previous = self.elapsed
        self.elapsed = min(self.duration, self.elapsed + dt)
        cumulative = 0.0
        for segment, duration in enumerate(self.durations):
            next_cumulative = cumulative + duration
            if self.elapsed <= next_cumulative or segment == len(self.durations) - 1:
                alpha = (self.elapsed - cumulative) / max(duration, 1e-9)
                data.qpos[self.joint_addrs] = self.points[segment] + (self.points[segment + 1] - self.points[segment]) * alpha
                data.qvel[self.dof_addrs] = 0.0
                break
            cumulative = next_cumulative
        cumulative = 0.0
        for point_index, duration in enumerate(self.durations, start=1):
            cumulative += duration
            if point_index not in self.fired and previous < cumulative <= self.elapsed + 1e-10:
                self.fired.add(point_index)
                for event in self.events:
                    if event.point_index == point_index:
                        event.callback()
        if self.elapsed >= self.duration - 1e-10:
            data.qpos[self.joint_addrs] = self.points[-1]
            data.qvel[self.dof_addrs] = 0.0
            self.complete = True
