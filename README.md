# XY Potential

Python implementation and optimisation of the two-dimensional XY model.

Each lattice site stores a planar spin angle. Nearest-neighbour bonds use
the XY interaction `-J cos(theta_i - theta_j)` with periodic boundaries.
The simulation visits every site once per sweep in a randomized order and
uses single-spin Metropolis proposals. Its order measure is the magnitude
of the mean planar spin vector, computed from the mean cosine and sine of
the angles.

## Environment

The project uses a Conda environment defined in `environment.yml`.

## Installation

Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) if it is not already available, then open a terminal in this project directory and create the project environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate xy-model
```

The environment includes Python 3.11, NumPy, Matplotlib, JupyterLab, and `ipykernel`. If you change the dependencies in `environment.yml` later, update the existing environment with:

```bash
conda env update -f environment.yml --prune
```

## Running from Jupyter

From the project directory, activate the environment and launch JupyterLab:

```bash
conda activate xy-model
jupyter lab
```

In JupyterLab, open `xy_experiments.ipynb`. Run its cells from top to bottom; the notebook imports the simulation functions from `xy_model.py`, so keep both files in the same project directory. Edit the settings in the experiment-settings cell (lattice sizes, temperatures, number of sweeps, and random seed), then rerun the subsequent cells to generate summaries and plots.
