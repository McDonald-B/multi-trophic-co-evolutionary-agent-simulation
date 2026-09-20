from __future__ import annotations
import numpy as np
from .config import Config
from .genome import Genome, GRAZER_TRAITS, PREDATOR_TRAITS
from .agents import Grazer, Predator, Corpse

GRAZER_IN, GRAZER_OUT = 8, 2
PREDATOR_IN, PREDATOR_OUT = 11, 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.step_i = 0
        self.plants = []  # list of [x, y]
        self.corpses: list[Corpse] = []
        self.grazers: list[Grazer] = []
        self.predators: list[Predator] = []
        self._bootstrap()

    # ---------------------------------------------------------- bootstrap
    def _bootstrap(self):
        cfg = self.cfg
        for _ in range(cfg.initial_plants):
            self.plants.append([self.rng.uniform(0, cfg.world_w), self.rng.uniform(0, cfg.world_h)])
        for _ in range(cfg.initial_grazers):
            g = Genome.random(GRAZER_TRAITS, GRAZER_IN, cfg.hidden_dim, GRAZER_OUT, self.rng)
            self.grazers.append(Grazer(self.rng.uniform(0, cfg.world_w), self.rng.uniform(0, cfg.world_h),
                                        g, cfg.max_energy_grazer * 0.6, rng=self.rng))
        for _ in range(cfg.initial_predators):
            g = Genome.random(PREDATOR_TRAITS, PREDATOR_IN, cfg.hidden_dim, PREDATOR_OUT, self.rng)
            self.predators.append(Predator(self.rng.uniform(0, cfg.world_w), self.rng.uniform(0, cfg.world_h),
                                            g, cfg.max_energy_predator * 0.6, rng=self.rng))

    # ---------------------------------------------------------- drought
    def drought_state(self):
        for start, dur, regrow_mult, cap_mult in self.cfg.drought_events:
            if start <= self.step_i < start + dur:
                return True, regrow_mult, cap_mult
        return False, 1.0, 1.0

    # ---------------------------------------------------------- plants
    def _update_plants(self, regrow_mult, cap_mult):
        cfg = self.cfg
        cap = int(cfg.max_plants * cap_mult)
        n = len(self.plants)
        if n < cap:
            spawn = int(round(n * cfg.plant_regrow_rate * regrow_mult)) if n > 0 else 4
            spawn = min(spawn, cap - n)
            for _ in range(max(spawn, 0)):
                if self.plants and self.rng.random() < 0.85:
                    bx, by = self.plants[self.rng.integers(0, len(self.plants))]
                    x = np.clip(bx + self.rng.normal(0, cfg.plant_cluster_jitter), 0, cfg.world_w)
                    y = np.clip(by + self.rng.normal(0, cfg.plant_cluster_jitter), 0, cfg.world_h)
                else:
                    x, y = self.rng.uniform(0, cfg.world_w), self.rng.uniform(0, cfg.world_h)
                self.plants.append([float(x), float(y)])

    # ---------------------------------------------------------- helpers
    @staticmethod
    def _nearest(pos, targets_xy):
        """pos: (2,) array. targets_xy: (N,2) array. Returns (idx, dx, dy, dist) or None."""
        if len(targets_xy) == 0:
            return None
        d = targets_xy - pos
        dist2 = np.einsum("ij,ij->i", d, d)
        idx = int(np.argmin(dist2))
        dist = float(np.sqrt(dist2[idx]))
        return idx, float(d[idx, 0]), float(d[idx, 1]), dist

    def _move(self, agent, turn_out, throttle_out, speed_trait_key, stamina_key, max_speed):
        cfg = self.cfg
        agent.heading += float(np.clip(turn_out, -1, 1)) * 0.6
        speed_mult = 0.5 + 0.9 * agent.traits[speed_trait_key]
        speed = float(np.clip(throttle_out, 0, 1)) * max_speed * speed_mult
        agent.x = float(np.clip(agent.x + np.cos(agent.heading) * speed, 0, cfg.world_w))
        agent.y = float(np.clip(agent.y + np.sin(agent.heading) * speed, 0, cfg.world_h))
        move_cost = cfg.move_metabolism_coeff * speed * (1.0 - 0.55 * agent.traits[stamina_key])
        return move_cost

    # ---------------------------------------------------------- main step
    def step(self):
        cfg = self.cfg
        drought_on, regrow_mult, cap_mult = self.drought_state()
        self._update_plants(regrow_mult, cap_mult)

        plants_xy = np.array(self.plants) if self.plants else np.zeros((0, 2))
        grazers_xy = np.array([[g.x, g.y] for g in self.grazers]) if self.grazers else np.zeros((0, 2))
        predators_xy = np.array([[p.x, p.y] for p in self.predators]) if self.predators else np.zeros((0, 2))
        corpses_xy = np.array([[c.x, c.y] for c in self.corpses]) if self.corpses else np.zeros((0, 2))

        eaten_plant_idx = set()
        grazer_detected_predator = [False] * len(self.grazers)
        grazer_nearest_plant = [None] * len(self.grazers)

        # density-dependent crowding pressure (keeps LV oscillations bounded, like a logistic term)
        ng, npd = len(self.grazers), len(self.predators)
        crowd_g = cfg.grazer_crowding_coeff * max(0.0, ng - cfg.grazer_soft_cap) / cfg.grazer_soft_cap
        crowd_p = cfg.predator_crowding_coeff * max(0.0, npd - cfg.predator_soft_cap) / cfg.predator_soft_cap

        # ---------------- Grazers: sense + decide + move
        for i, g in enumerate(self.grazers):
            pos = np.array([g.x, g.y])
            detect_range, half_angle = g.cone_params(cfg.grazer_detect_base, cfg.grazer_detect_range_scale)

            plant_near = self._nearest(pos, plants_xy)
            p_dx = p_dy = p_dinv = 0.0
            if plant_near is not None:
                idx, dx, dy, dist = plant_near
                if dist < detect_range * 1.4:  # plants are easy to spot (no camouflage)
                    p_dx, p_dy = dx / 250.0, dy / 250.0
                    p_dinv = max(0.0, 1.0 - dist / (detect_range * 1.4))
                grazer_nearest_plant[i] = (idx, dist)

            pr_dx = pr_dy = pr_dinv = 0.0
            pred_near = self._nearest(pos, predators_xy)
            if pred_near is not None:
                idx, dx, dy, dist = pred_near
                pred_camo = self.predators[idx].traits["camouflage"]
                if g.sees(dx, dy, dist, detect_range, half_angle, target_camouflage=pred_camo):
                    pr_dx, pr_dy = dx / 250.0, dy / 250.0
                    pr_dinv = max(0.0, 1.0 - dist / detect_range)
                    grazer_detected_predator[i] = True

            energy_frac = g.energy / cfg.max_energy_grazer
            x_in = np.array([p_dx, p_dy, p_dinv, pr_dx, pr_dy, pr_dinv, energy_frac, 1.0])
            out = g.genome.brain.forward(x_in)
            turn_out, throttle_out = np.tanh(out[0]), sigmoid(out[1])
            # baseline foraging instinct: bias toward sensed food (NN still modulates the rest)
            if p_dinv > 0:
                toward = np.arctan2(p_dy, p_dx)
                diff = np.arctan2(np.sin(toward - g.heading), np.cos(toward - g.heading))
                turn_out = float(np.clip(turn_out + 0.55 * np.sign(diff), -1, 1))
                throttle_out = max(throttle_out, 0.4)
            # flee override: strong pull directly away from a detected predator (dominates foraging)
            if grazer_detected_predator[i]:
                away = np.arctan2(-pr_dy, -pr_dx)
                diff = np.arctan2(np.sin(away - g.heading), np.cos(away - g.heading))
                turn_out = float(np.clip(turn_out + 0.7 * np.sign(diff), -1, 1))
                throttle_out = max(throttle_out, 0.75)

            move_cost = self._move(g, turn_out, throttle_out, "burst_speed", "stamina", cfg.max_speed)
            g.energy -= cfg.base_metabolism + g.genome.metabolic_cost(cfg.trait_metabolism_coeff * 0.5) + move_cost + crowd_g
            g.age += 1

        # ---------------- Eating
        for i, g in enumerate(self.grazers):
            near = grazer_nearest_plant[i]
            if near is None:
                continue
            idx, dist = near
            if dist <= cfg.plant_eat_radius and idx not in eaten_plant_idx:
                eaten_plant_idx.add(idx)
                g.energy = min(cfg.max_energy_grazer, g.energy + min(cfg.plant_energy, cfg.grazer_eat_gain_cap))
        if eaten_plant_idx:
            self.plants = [p for j, p in enumerate(self.plants) if j not in eaten_plant_idx]

        # ---------------- Predators: sense + decide + move
        grazers_xy = np.array([[g.x, g.y] for g in self.grazers]) if self.grazers else np.zeros((0, 2))
        pred_ally_targets = predators_xy
        killed_grazers = set()
        for pi, pr in enumerate(self.predators):
            pos = np.array([pr.x, pr.y])
            detect_range, half_angle = pr.cone_params(cfg.predator_detect_base, cfg.predator_detect_range_scale)

            prey_dx = prey_dy = prey_dinv = 0.0
            prey_near = self._nearest(pos, grazers_xy)
            prey_detected = False
            target_idx, target_dist = None, None
            if prey_near is not None:
                idx, dx, dy, dist = prey_near
                prey_camo = self.grazers[idx].traits["camouflage"]
                if pr.sees(dx, dy, dist, detect_range, half_angle, target_camouflage=prey_camo):
                    prey_dx, prey_dy = dx / 250.0, dy / 250.0
                    prey_dinv = max(0.0, 1.0 - dist / detect_range)
                    prey_detected = True
                    target_idx, target_dist = idx, dist

            ally_dx = ally_dy = ally_dinv = 0.0
            others = np.delete(pred_ally_targets, pi, axis=0) if len(pred_ally_targets) > 1 else np.zeros((0, 2))
            ally_near = self._nearest(pos, others)
            if ally_near is not None:
                idx, dx, dy, dist = ally_near
                if dist < cfg.pack_bonus_radius * 2.5:
                    ally_dx, ally_dy = dx / 250.0, dy / 250.0
                    ally_dinv = max(0.0, 1.0 - dist / (cfg.pack_bonus_radius * 2.5))

            corpse_dx = corpse_dy = corpse_dinv = 0.0
            corpse_near = self._nearest(pos, corpses_xy)
            if corpse_near is not None:
                idx, dx, dy, dist = corpse_near
                if dist < detect_range * 1.2:
                    corpse_dx, corpse_dy = dx / 250.0, dy / 250.0
                    corpse_dinv = max(0.0, 1.0 - dist / (detect_range * 1.2))

            energy_frac = pr.energy / cfg.max_energy_predator
            x_in = np.array([prey_dx, prey_dy, prey_dinv, ally_dx, ally_dy, ally_dinv,
                              corpse_dx, corpse_dy, corpse_dinv, energy_frac, 1.0])
            out = pr.genome.brain.forward(x_in)
            turn_out, throttle_out = np.tanh(out[0]), sigmoid(out[1])
            if prey_detected:
                toward = np.arctan2(prey_dy, prey_dx)
                diff = np.arctan2(np.sin(toward - pr.heading), np.cos(toward - pr.heading))
                turn_out = float(np.clip(turn_out + 0.7 * np.sign(diff), -1, 1))
                throttle_out = max(throttle_out, 0.7)

            move_cost = self._move(pr, turn_out, throttle_out, "pursuit_speed", "stamina", cfg.max_speed * 1.05)
            pr.energy -= cfg.predator_base_metabolism + pr.genome.metabolic_cost(cfg.trait_metabolism_coeff) + move_cost + crowd_p
            pr.age += 1

            # ---- attack attempt
            engage_radius = cfg.predator_kill_radius * cfg.predator_engage_mult
            if target_idx is not None and target_idx not in killed_grazers and target_dist < engage_radius:
                grazer_target = self.grazers[target_idx]
                ally_count = int(np.sum(np.linalg.norm(others - pos, axis=1) < cfg.pack_bonus_radius)) if len(others) else 0
                pack_bonus = min(cfg.pack_bonus_cap, ally_count * cfg.pack_bonus_per_ally * pr.traits["pack_affinity"])
                ambush_bonus = 0.0
                if not grazer_detected_predator[target_idx] and pr.traits["ambush"] > 0.5:
                    ambush_bonus = 0.22 * pr.traits["ambush"]
                speed_edge = pr.traits["pursuit_speed"] - grazer_target.traits["burst_speed"]
                evasion_penalty = cfg.attack_evasion_penalty if grazer_detected_predator[target_idx] else 0.0
                p_success = sigmoid(speed_edge * 3.0) * cfg.attack_base_scale + pack_bonus + ambush_bonus - evasion_penalty
                p_success = float(np.clip(p_success, 0.02, 0.85))
                if self.rng.random() < p_success:
                    killed_grazers.add(target_idx)
                    pr.energy = min(cfg.max_energy_predator, pr.energy + cfg.predator_energy_from_kill)
                    pr.kills += 1

            # ---- scavenging
            if corpse_near is not None:
                idx, dx, dy, dist = corpse_near
                if dist <= cfg.plant_eat_radius * 1.5 and idx < len(self.corpses):
                    c = self.corpses[idx]
                    if c.energy > 0:
                        gain = min(cfg.predator_energy_from_corpse, c.energy)
                        pr.energy = min(cfg.max_energy_predator, pr.energy + gain)
                        c.energy -= gain

        # ---------------- resolve grazer deaths (predation)
        new_grazers = []
        for i, g in enumerate(self.grazers):
            if i in killed_grazers:
                continue
            new_grazers.append(g)
        self.grazers = new_grazers

        # ---------------- natural deaths -> corpses; reproduction
        self._resolve_deaths_and_reproduction()

        # ---------------- corpse decay
        for c in self.corpses:
            c.age += 1
        self.corpses = [c for c in self.corpses if c.age < cfg.corpse_lifetime and c.energy > 0.05]

        self.step_i += 1
        return drought_on

    def _resolve_deaths_and_reproduction(self):
        cfg = self.cfg
        rng = self.rng

        survivors = []
        for g in self.grazers:
            if g.energy <= 0 or g.age >= cfg.max_age:
                self.corpses.append(Corpse(g.x, g.y, cfg.plant_energy * 1.5))
                continue
            survivors.append(g)
        self.grazers = survivors

        offspring = []
        for g in self.grazers:
            if g.energy >= cfg.reproduce_energy_frac_grazer * cfg.max_energy_grazer and len(self.grazers) + len(offspring) < 2000:
                child_energy = g.energy * cfg.reproduce_cost_frac
                g.energy -= child_energy
                child_genome = g.genome.clone_mutated(cfg, rng)
                cx = np.clip(g.x + rng.normal(0, 10), 0, cfg.world_w)
                cy = np.clip(g.y + rng.normal(0, 10), 0, cfg.world_h)
                offspring.append(Grazer(cx, cy, child_genome, child_energy, g.generation + 1, g.id, rng=rng))
        self.grazers.extend(offspring)

        survivors_p = []
        for p in self.predators:
            if p.energy <= 0 or p.age >= cfg.max_age:
                continue  # predators don't leave (edible) corpses in this model
            survivors_p.append(p)
        self.predators = survivors_p

        offspring_p = []
        for p in self.predators:
            if p.energy >= cfg.reproduce_energy_frac_predator * cfg.max_energy_predator and len(self.predators) + len(offspring_p) < 500:
                child_energy = p.energy * cfg.reproduce_cost_frac
                p.energy -= child_energy
                child_genome = p.genome.clone_mutated(cfg, rng)
                cx = np.clip(p.x + rng.normal(0, 10), 0, cfg.world_w)
                cy = np.clip(p.y + rng.normal(0, 10), 0, cfg.world_h)
                offspring_p.append(Predator(cx, cy, child_genome, child_energy, p.generation + 1, p.id, rng=rng))
        self.predators.extend(offspring_p)
