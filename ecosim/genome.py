"""
Genome = (Brain weights, Scalar traits).

Brain: a tiny feed-forward NN (evolved by mutation, not gradient descent -
this is neuroevolution). It maps sensory input -> (turn, throttle).

Scalar traits: physical capabilities (camouflage, acuity, stamina, sensory
cone shape, pack affinity, ambush tendency...). Every trait has a metabolic
upkeep cost, so tradeoffs between traits emerge from energy budget pressure
rather than being hard-coded - e.g. an agent that invests heavily in both
camouflage AND acuity will starve faster than one that specializes.
"""
from __future__ import annotations
import numpy as np


class Brain:
    """Single-hidden-layer MLP. Weights are the evolvable genome."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rng: np.random.Generator, weights=None):
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim
        if weights is None:
            scale1 = np.sqrt(2.0 / in_dim)
            scale2 = np.sqrt(2.0 / hidden_dim)
            self.W1 = rng.normal(0, scale1, size=(in_dim, hidden_dim))
            self.b1 = np.zeros(hidden_dim)
            self.W2 = rng.normal(0, scale2, size=(hidden_dim, out_dim))
            self.b2 = np.zeros(out_dim)
        else:
            self.W1, self.b1, self.W2, self.b2 = weights

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        o = h @ self.W2 + self.b2
        return o

    def clone_mutated(self, rng: np.random.Generator, sigma: float, big_jump_prob: float, big_sigma: float) -> "Brain":
        def mutate(arr):
            noise = rng.normal(0, sigma, size=arr.shape)
            if rng.random() < big_jump_prob:
                noise = noise + rng.normal(0, big_sigma, size=arr.shape)
            return arr + noise
        weights = (mutate(self.W1), mutate(self.b1), mutate(self.W2), mutate(self.b2))
        return Brain(self.in_dim, self.hidden_dim, self.out_dim, rng, weights=weights)

    def flat_summary(self):
        # cheap scalar fingerprint used only for display/debugging
        return float(np.mean(np.abs(self.W1))) + float(np.mean(np.abs(self.W2)))


GRAZER_TRAITS = [
    "camouflage",      # reduces detectability by predators
    "acuity",          # improves own detection range of plants & predators
    "burst_speed",     # top sprint speed multiplier
    "stamina",         # reduces metabolic drain while moving fast
    "cone_width",       # wide sensory cone: sees more directions...
    "cone_depth",       # ...vs narrow-but-far cone: sees further in one direction
]

PREDATOR_TRAITS = [
    "pursuit_speed",
    "stamina",
    "acuity",
    "camouflage",       # ambush camo — prey harder to detect the predator
    "pack_affinity",    # tendency to coordinate / benefit from nearby allies
    "ambush",           # 0 = active pursuit style, 1 = sit-and-wait ambush style
]


def random_traits(names, rng: np.random.Generator) -> dict:
    return {n: float(rng.uniform(0.25, 0.75)) for n in names}


def mutate_traits(traits: dict, rng: np.random.Generator, sigma: float, big_jump_prob: float, big_sigma: float) -> dict:
    out = {}
    for k, v in traits.items():
        noise = rng.normal(0, sigma)
        if rng.random() < big_jump_prob:
            noise += rng.normal(0, big_sigma)
        out[k] = float(np.clip(v + noise, 0.0, 1.0))
    return out


class Genome:
    __slots__ = ("brain", "traits")

    def __init__(self, brain: Brain, traits: dict):
        self.brain = brain
        self.traits = traits

    @staticmethod
    def random(trait_names, in_dim, hidden_dim, out_dim, rng):
        return Genome(Brain(in_dim, hidden_dim, out_dim, rng), random_traits(trait_names, rng))

    def clone_mutated(self, cfg, rng) -> "Genome":
        return Genome(
            self.brain.clone_mutated(rng, cfg.mutation_sigma_brain, cfg.mutation_big_jump_prob, cfg.mutation_big_jump_sigma),
            mutate_traits(self.traits, rng, cfg.mutation_sigma_trait, cfg.mutation_big_jump_prob, cfg.mutation_big_jump_sigma),
        )

    def metabolic_cost(self, coeff: float) -> float:
        return coeff * sum(self.traits.values())
