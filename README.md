# MCMC Optimization for Circular Permutations 

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/qqaazz800624/MCMC_circular_data)

**Repository:** [https://github.com/qqaazz800624/MCMC_circular_data](https://github.com/qqaazz800624/MCMC_circular_data)

Markov Chain Monte Carlo (MCMC) algorithms for optimizing circular data and permutation spaces, featuring topology-aware proposal distributions like 2-opt and 3-opt variants.

## Overview

This repository provides a robust, modular, and highly vectorized framework for solving complex combinatorial optimization problems in circular permutation spaces ($S_n$). It is particularly designed for objective functions that feature a combination of **absolute positional weights (Linear Term)** and **circular topological interactions (Product/Wrap-around Term)**.

Potential applications span across multiple domains:
- **Sports Analytics:** Optimizing batting orders based on player interaction and base-running topology.
- **Manufacturing:** Minimizing tool-switching time in Rotary Tool Magazines (CNC machines).
- **Bioinformatics & Genomics:** Analyzing sequence alignments and evolutionary genome rearrangements in circular DNA structures (e.g., bacterial plasmids and mitochondrial genomes).
- **Computer Vision:** Circular bounding box matching and periodic boundary condition evaluations.

## Key Features

- **Modular Architecture:** Objective functions and proposal distributions are decoupled, allowing researchers to easily plug in new $F(x)$ or $J(x'|x)$ functions without altering the core MCMC engine.
- **Topology-Aware Proposals:** Implements advanced transition kernels for circular data:
  - Random Swap (Symmetric baseline)
  - Directional Block Reversal (2-opt variant)
  - Adjacent Block Exchange (3-opt variant)
- **"God-Mode" Landscape Analysis:** Includes a highly optimized script to exhaustively enumerate all $n!$ states (for $n \le 10$), calculate the true Boltzmann distribution, and visualize the objective landscape (CDFs).
- **Variance-Balanced Objective Functions:** Built-in statistical tools to dynamically calculate the standard deviation of sub-terms, ensuring that $\alpha$ and $\beta$ weights perfectly balance the energy landscape to avoid "term dominance."

## Requirements & Installation

**Important:** We highly recommend using **Python 3.12** for this project to ensure perfect compatibility with the underlying scientific computing libraries (e.g., NumPy, Matplotlib, IPython).

It is recommended to set up a clean virtual environment using Conda:

```bash
# 1. Create a new environment with Python 3.12
conda create -n mcmc_env python=3.12 -y

# 2. Activate the environment
conda activate mcmc_env

# 3. Clone the repository and install dependencies
git clone [https://github.com/qqaazz800624/MCMC_circular_data.git](https://github.com/qqaazz800624/MCMC_circular_data.git)
cd MCMC_circular_data
pip install -r requirements.txt