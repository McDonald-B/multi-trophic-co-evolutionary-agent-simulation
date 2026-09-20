from __future__ import annotations
import numpy as np
from .genome import Genome

_next_id = [0]


def _new_id():
    _next_id[0] += 1
    return _next_id[0]


class Corpse:
    __slots__ = ("x", "y", "age", "energy")

    def __init__(self, x, y, energy):
        self.x, self.y, self.age, self.energy = x, y, 0, energy


class Agent:
    """Base class for Grazer / Predator."""

    def __init__(self, x, y, genome: Genome, energy, generation=0, parent_id=None, rng=None):
        self.id = _new_id()
        self.x, self.y = x, y
        rng = rng if rng is not None else np.random.default_rng()
        self.heading = rng.uniform(-np.pi, np.pi)
        self.genome = genome
        self.energy = energy
        self.age = 0
        self.generation = generation
        self.parent_id = parent_id
        self.alive = True
        self.kills = 0

    @property
    def traits(self):
        return self.genome.traits

    def cone_params(self, base_range, range_scale):
        t = self.traits
        acuity = t.get("acuity", 0.5)
        cone_width = t.get("cone_width", 0.5)
        cone_depth = t.get("cone_depth", 0.5)
        detect_range = (base_range + range_scale * acuity) * (0.55 + 0.85 * cone_depth)
        half_angle = (0.22 + 0.78 * cone_width) * np.pi  # wide cone_width -> near-omnidirectional
        return detect_range, half_angle

    def sees(self, dx, dy, dist, detect_range, half_angle, target_camouflage=0.0):
        if dist < 1e-6:
            return True
        visibility = dist * (1.0 + 1.6 * target_camouflage)
        if visibility > detect_range:
            return False
        ang = np.arctan2(dy, dx)
        diff = np.abs(np.arctan2(np.sin(ang - self.heading), np.cos(ang - self.heading)))
        return diff <= half_angle


class Grazer(Agent):
    kind = "grazer"


class Predator(Agent):
    kind = "predator"
