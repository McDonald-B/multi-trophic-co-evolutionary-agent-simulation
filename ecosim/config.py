"""
Central configuration for the multi-trophic food web simulation.
All tunable parameters live here so experiments are reproducible and
easy to sweep.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- World ---
    world_w: float = 1000.0
    world_h: float = 700.0
    seed: int = 42

    # --- Plants (vegetation) ---
    max_plants: int = 500
    initial_plants: int = 260
    plant_regrow_rate: float = 0.06       # fraction of current pop that spawns new plants/step
    plant_energy: float = 6.0             # energy a grazer gets from eating one plant
    plant_eat_radius: float = 6.0
    plant_cluster_jitter: float = 28.0    # new plants spawn near existing ones (clustering)

    # --- Population bootstrapping ---
    initial_grazers: int = 90
    initial_predators: int = 16

    # --- Shared agent physics ---
    max_speed: float = 3.2                # world units / step at throttle=1, trait=1
    base_metabolism: float = 0.05         # passive energy drain / step (grazer)
    predator_base_metabolism: float = 0.11  # carnivores are energetically expensive -> naturally rarer
    trait_metabolism_coeff: float = 0.035 # extra drain per unit of invested trait
    move_metabolism_coeff: float = 0.02   # energy cost scaling with speed actually used
    max_age: int = 1400
    reproduce_energy_frac_grazer: float = 0.70    # frac of max_energy needed to reproduce
    reproduce_energy_frac_predator: float = 0.90  # predators need much more surplus to reproduce
    reproduce_cost_frac: float = 0.5      # frac of energy given to offspring
    max_energy_grazer: float = 40.0
    max_energy_predator: float = 70.0
    mutation_sigma_trait: float = 0.06
    mutation_sigma_brain: float = 0.15
    mutation_big_jump_prob: float = 0.04
    mutation_big_jump_sigma: float = 0.35

    # --- Grazer specifics ---
    grazer_eat_gain_cap: float = 6.0
    grazer_detect_base: float = 90.0      # base sensory range (before acuity trait scaling)
    grazer_detect_range_scale: float = 140.0

    # --- Predator specifics ---
    predator_kill_radius: float = 7.0
    predator_engage_mult: float = 1.9           # engage radius = kill_radius * this
    predator_detect_base: float = 70.0
    predator_detect_range_scale: float = 160.0
    predator_energy_from_kill: float = 15.0
    predator_energy_from_corpse: float = 10.0   # scavenging is easier but lower payoff
    corpse_lifetime: int = 300                  # steps before a corpse decays away
    grazer_soft_cap: int = 420                  # crowding pressure kicks in above this pop
    grazer_crowding_coeff: float = 0.16
    predator_soft_cap: int = 170
    predator_crowding_coeff: float = 0.22
    pack_bonus_radius: float = 55.0             # allies within this range boost attack success
    pack_bonus_per_ally: float = 0.09
    pack_bonus_cap: float = 0.35
    ambush_burst_multiplier: float = 2.1        # bonus effective speed on first strike if undetected
    attack_base_scale: float = 0.32             # scales speed-edge sigmoid into base attack prob
    attack_evasion_penalty: float = 0.30        # prob penalty if prey detected predator in advance

    # --- Neural brain architecture ---
    hidden_dim: int = 12

    # --- Drought shock event(s) ---
    # list of (start_step, duration, regrow_multiplier, max_plants_multiplier)
    drought_events: list = field(default_factory=lambda: [(1200, 300, 0.35, 0.55)])

    # --- Simulation run ---
    n_steps: int = 3000
    snapshot_every: int = 15     # store a full agent snapshot every N steps (for playback)
    log_every: int = 1           # population/trait aggregate logging cadence
