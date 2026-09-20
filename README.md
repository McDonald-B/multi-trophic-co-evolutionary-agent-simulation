# Multi-Trophic Co-Evolutionary Agent Simulation

An artificial-life simulation exploring predator-prey dynamics and evolution in a three-level food web:

**Vegetation → Herbivores → Predators**

The project was built from scratch in Python. Agents compete for resources, hunt, reproduce and die within a shared environment. Herbivores and predators use small neural networks alongside heritable traits, with successful agents passing their characteristics to the next generation through mutation.

The simulation produces structured population and trait data which is then processed into an interactive dashboard for exploring the results.

![Dashboard screenshot](docs/dashboard_screenshot.gif)

## Demo

The project includes a desktop dashboard where a simulation can be replayed and new experiments can be generated.

The dashboard includes:

* Population trends
* Trait evolution
* Herbivore and predator populations
* Drought events
* Recorded agent movement
* Adjustable playback speed
* Configurable random seed
* Configurable simulation length
* Live progress while generating a new simulation

The ecosystem itself is rendered on a canvas, with different shapes used for herbivores and predators so that interactions are easier to follow.

## How it works

Each animal has two main components:

### Heritable traits

Agents have traits that affect how they interact with the environment, including sensory ability, movement, camouflage, stamina and behavioural tendencies.

Traits are passed to offspring with mutation.

These traits also have associated energy costs, creating trade-offs between capability and survival.

### Neural networks

Each animal has a small feed-forward neural network which receives information about its surroundings and influences movement decisions.

Inputs include information such as:

* Nearby food
* Nearby prey
* Nearby predators
* Nearby members of the same species
* Agent energy

The network produces movement decisions such as turning and throttle.

A small instinct layer provides basic behaviours such as moving towards detected food and avoiding predators. This gives newly-created agents enough basic behaviour to survive while their neural-network parameters evolve.

## Ecosystem

The simulation contains three main trophic levels.

```text
Vegetation
    ↓
Herbivores
    ↓
Predators
```

Herbivores consume vegetation while predators hunt herbivores.

Predators can also consume corpses left by animals that die from causes such as starvation. This provides another source of energy when live prey becomes scarce.

Population growth is constrained by resource availability, energy expenditure and density-dependent pressure.

## Environmental events

The simulation supports configurable environmental shocks.

For example, the default configuration includes a drought which temporarily reduces vegetation regeneration and carrying capacity.

This creates a useful experiment for observing how changes at the bottom of the food web propagate through herbivore and predator populations.

Drought events are configured in `ecosim/config.py`:

```python
drought_events = [
    (1200, 300, 0.35, 0.55)
]
```

The values represent:

```text
(start step, duration, plant regrowth multiplier, plant capacity multiplier)
```

## Data pipeline

One of the main goals of the project was to separate the simulation from the analysis and visualisation layers.

```text
Python simulation
       │
       ├── population data
       ├── trait data
       └── agent snapshots
                │
                ▼
        CSV / JSON files
                │
                ▼
       build_dashboard.py
                │
                ▼
   self-contained HTML dashboard
```

The simulation records data at each step and periodically records complete snapshots of the ecosystem.

`build_dashboard.py` processes these outputs and embeds the required data directly into the dashboard template.

The resulting HTML file can therefore be opened without a backend or database.

## Project structure

```text
Multi-Trophic Co-Evolutionary Agent Simulation/
│
├── ecosim/
│   ├── config.py
│   ├── genome.py
│   ├── agents.py
│   ├── world.py
│   └── simulate.py
│
├── build_dashboard.py
├── dashboard_template.html
├── run.py
├── requirements.txt
└── README.md
```

### Key components

**`config.py`**

Central configuration for population sizes, energy costs, mutation rates, environmental events and other simulation parameters.

**`genome.py`**

Contains the neural-network implementation, heritable traits and mutation logic.

**`agents.py`**

Defines the different agent types and their behaviour.

**`world.py`**

Runs the ecosystem itself, including sensing, movement, feeding, hunting, reproduction and environmental events.

**`simulate.py`**

Runs an experiment and writes the resulting datasets to disk.

**`build_dashboard.py`**

Processes simulation output and generates the interactive dashboard.

**`dashboard_template.html`**

Frontend for the visualisation and recorded ecosystem playback.

**`run.py`**

Provides the desktop application wrapper and allows new simulations to be generated directly from the dashboard.

## Running the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python run.py
```

The application will generate an initial simulation and open the dashboard.

New simulations can then be started from the dashboard by changing the seed and/or number of steps.

`run.py` uses [pywebview](https://pywebview.flowrl.com/) to open the dashboard in a native window and auto-detects the right rendering backend for your OS (WebView2 on Windows, WebKit on macOS, GTK/QT on Linux) — no platform-specific setup needed.

Prefer the command line instead? The desktop app just automates these two commands, which work identically on their own:

```bash
python -m ecosim.simulate --steps 3000 --seed 42 --out run
python build_dashboard.py --prefix run --out ecosystem_dashboard.html
```

Then open `ecosystem_dashboard.html` directly in any browser — no server needed.

## Simulation output

Each run produces several files.

### `run_population.csv`

Population counts recorded throughout the simulation, including:

* Plants
* Herbivores
* Predators
* Corpses

### `run_traits.csv`

Population-level statistics for the heritable traits of each agent type.

### `run_snapshots.json`

Periodic snapshots of the ecosystem used for the recorded canvas playback.

### `run_summary.json`

Summary information and metadata for the simulation run.

> These four files (plus `ecosystem_dashboard.html`) are regenerated on every run and excluded via `.gitignore` — they aren't committed to the repo. Run the commands above to produce your own.

## Reproducibility

The simulation uses a configurable random seed.

For example:

```text
Seed: 42
Steps: 3000
```

Using the same seed and configuration allows the same simulation to be reproduced.

Different seeds can be used to explore how much variation exists between evolutionary runs.

This also makes it possible to run controlled experiments where one parameter is changed while keeping the initial random state consistent.

## Configuration

The main parameters are contained in:

```text
ecosim/config.py
```

Some of the parameters that can be experimented with include:

```text
initial_grazers
initial_predators
mutation_sigma_trait
mutation_sigma_brain
pack_bonus_per_ally
pack_bonus_radius
predator_energy_from_corpse
drought_events
```

Changing these parameters can produce substantially different population dynamics and evolutionary outcomes.

## Dashboard performance

Because the simulation can generate thousands of population records and hundreds of agent snapshots, `build_dashboard.py` keeps the generated HTML file manageable automatically: population and trait rows are downsampled to a target row count (1,500 and 5,000 rows respectively) rather than embedding every single logged step.

```bash
python build_dashboard.py \
    --prefix run \
    --out ecosystem_dashboard.html
```

Playback snapshots (used for the canvas replay) are **not** downsampled — every frame recorded by `simulate.py` is embedded, since the replay needs the actual spatial states. Snapshot frequency is controlled at simulation time instead, via `snapshot_every` in `ecosim/config.py` (default: every 15 steps). If you push `--steps` much higher than the 3,000-step default, keep an eye on the printed output file size — increase `snapshot_every` if it gets unwieldy.

## What I wanted to explore

The project started from a simple question:

> What happens when population dynamics and individual behaviour are allowed to evolve together?

Rather than defining a fixed predator strategy or manually scripting how animals should behave, the simulation gives agents a set of capabilities, costs and sensory inputs and allows selection to act on them over time.

This makes the resulting data useful for exploring questions around:

* Population stability
* Predator-prey cycles
* Resource scarcity
* Trait selection
* Behavioural adaptation
* Recovery from environmental shocks

## Possible next steps

Some areas I would like to explore further:

* Track and visualise evolutionary lineages.
* Analyse neural-network weights of successful agents.
* Run large batches of simulations and compare outcomes statistically.
* Introduce different predator strategies and compare their population dynamics.
* Add more environmental events and resource types.
* Build automated experiment and parameter-sweep tooling.

## Technologies

* **Python**
* **NumPy**
* **JavaScript**
* **HTML / CSS**
* **Canvas API**
* **pywebview**
* **CSV / JSON**
* **Git**

## License

MIT — see [LICENSE](LICENSE).

## Author

Built as a personal software engineering and data project, with a focus on combining simulation, evolutionary algorithms, data processing and interactive visualisation.
