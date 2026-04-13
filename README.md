# MCMC Optimization for Circular Permutations

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

## Repository Structure

```text
MCMC_circular_data/
├── main_simulation.py      # Entry point for MCMC simulations (CLI supported)
├── analyze_landscape.py    # God-mode exhaustive search & Variance balancing
├── objectives.py           # Definitions of objective functions F(x)
├── proposals.py            # Proposal distributions J(x'|x)
├── utils.py                # Plotting tools (CDF, Trace plots) and helpers
└── README.md               # Project documentation