# Course Scheduling Prototype

This project explores how genetic algorithms (GA) and a multi-objective NSGA-II approach can generate feasible timetables for a university curriculum. Each chromosome represents a schedule matrix where rows correspond to time slots and columns correspond to rooms.

## Features

- **Single-objective GA** with basic crossover, mutation, and constraint checking.
- **NSGA-II extension** to evolve a population toward a Pareto front balancing:
  - Average distance between rooms.
  - Average room size.
- Utilities for exporting/importing chromosomes and analyzing results.

## Requirements

- Python 3.8+
- Packages: `numpy`, `pandas`, `matplotlib`

```bash
pip install numpy pandas matplotlib
```

*You may also create a **`requirements.txt`** with these packages.*

## Running an Experiment

Ensure the `csv/` files (`rooms.csv`, `subjects.csv`, `curriculum.csv`) are available.

```bash
python main.py
```

The script randomly samples crossover and mutation rates, runs the NSGA-II algorithm for each configuration, and saves results under `simulation/`. View generated figures in `simulation/run_X/` and evaluation plots in `fig/`.

## Collecting Pareto Fronts

Once multiple runs complete, gather the resulting Pareto fronts:

```bash
python collect_pareto_fronts.py
```

This script copies Pareto front files from `simulation/run_*` into `all_pareto_fronts/`.

## Analyzing Results

`analyze.py` can visualize the combined Pareto fronts:

```bash
python analyze.py
```

This script loads chromosomes from `all_pareto_fronts/`, initializes an NSGA-II instance, and plots the objective space.

## Configuration

Algorithm parameters reside in `globals.py`:

- Population size
- Mutation/crossover rates
- Mutation points
- Evaluation method (`AVERAGE_DISTANCE` or `AVERAGE_SIZE`)
- Flags for single- vs. multi-objective mode

Editing this file allows quick experimentation with different settings.

## Notes

This repository is a prototype and may require further tuning for large datasets or more complex constraints. Contributions are welcome!
