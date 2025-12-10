<p align="center">
  <img src="logo.png" width="300" alt="Logo" />
</p>

# Scalable h-adaptive probabilistic solver for time-independent and time-dependent systems

This repository contains the implementation for the paper:

> **Scalable h-adaptive probabilistic solver for time-independent and time-dependent systems**  

## Overview
The code implements a scalable **Gaussian process probabilistic numerical solver (GPPS)** for partial differential equations that combines a Gaussian process (GP) representation of the solution, a stochastic dual descent (SDD) algorithm for fast inference, and a **clustering-based active learning** strategy for **h-adaptive** refinement. The method applies to both time-independent (steady-state) and time-dependent (space–time) PDEs, and returns not only numerical solutions but also rigorous posterior uncertainty quantification (UQ). Unlike standard GP-based solvers, the approach in the paper is computationally scalable, and high-dimensional problems.

## Example

**Case Study 2: Poisson equation in 3D domain**

![fig](image.png)

---
## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/savnkr/GPPS.git
    cd GPPS
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

## Requirements

To run the code, ensure that you have the following dependencies installed:

- Python 3.12
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Other libraries specified in `requirements.txt`

## Citation

If you use this code in your research, please cite the following paper:
```bibtex
@misc{thakur2025scalablehadaptiveprobabilisticsolver,
      title={Scalable h-adaptive probabilistic solver for time-independent and time-dependent systems}, 
      author={Akshay Thakur and Sawan Kumar and Matthew Zahr and Souvik Chakraborty},
      year={2025},
      eprint={2508.09623},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2508.09623}, 
}
```
