# P3MLens
An Accurate P3M Algorithm for Gravitational Lensing Studies in Simulations.

The code of current version is parallelized using numba based on threads, which may not be suitable for large cosmological simulations. New version based on mpi4py is under development.

P3MLens construct a lens plane from input particles and return the quantities like $\bm{E}$, $\bm{\alpha}$, $\kappa$, $\gamma$ and $\mu$ for positions of interest.

## Installation
The easiest way to install the P3MLens module is to use pip:

`pip install P3MLens `

## Dependencies
This implementation of P3MLens depends on:

* numpy>=1.10
* scipy>=1.6.0
* astropy>=4.0
* numba>=0.50.0

## Usage
`from P3MLens.Plane import Plane`
