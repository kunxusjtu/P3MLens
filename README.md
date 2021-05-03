# P3MLens
An Accurate P3M Algorithm for Gravitational Lensing Studies in Simulations.

The code of current version is parallelized using numba based on threads, which may not be suitable for large cosmological simulations. New version based on mpi4py is under development.

P3MLens construct a lens plane from input particles and return the quantities like $\bm{E}$, $\bm{\alpha}$, $\kappa$, $\gamma$ and $\mu$ for positions of interest.

## Installation
The easiest way to install the P3MLens module is to use pip:

`$ pip install P3MLens `

## Dependencies
This implementation of P3MLens depends on:

* numpy>=1.10
* scipy>=1.6.0
* astropy>=4.0
* numba>=0.50.0

## Usage
P3MLens.Plane contains a class named Plane, which is the main part of the algorithm, and a function named Green, which is used to calculate the 2D optimized Green funciton. With the 2D positions `coor` of the particles, a lens plane can be constructed: 

`from P3MLens.Plane import Plane`  
`lens = Plane(coor, box=1000, m_p=5e2)`
