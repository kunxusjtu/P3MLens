# P3MLens
An Accurate P3M Algorithm for Gravitational Lensing Studies in Simulations.

The code of current version is parallelized using `numba` based on threads, which may not be suitable for large cosmological simulations. New version based on `mpi4py` is under development.

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
* pyFFTW (optional)

## Usage
`P3MLens.Plane` contains a class named `Plane`, which is the main part of the algorithm, and a function named `Green`, which is used to calculate the 2D optimized Green funciton. With the 2D positions `coor` of the particles, a lens plane can be constructed: 

```python
from P3MLens.Plane import Plane  
lens = Plane(coor, box=1000, m_p=5e2)
```
Then, PM, PP and total force field for positions of interest can be calculated with the input `x` and `y`:

```python
PM = lens.PM_field(x, y)
PP = lens.PP_field(x, y)
P3M = lens.total_field(x, y)
```

Deflection angles and lensing parameters can be gotten in the similar way:

```python
angles = lens.deflection_angle(x, y)
convergence, shear1, shear2, magnification = lens.lense_parameter(x, y, zl=0.5, zs=1.0)
```

If you want to construct many `Plane`s with the same configuration (box size, grid size, mass assignment and smoothing), you're recommanded to calculate the optimized Green function by yourself and input it to the `Plane` class. So the `Plane` class won't waste time to calculate the Green function for every `Plane`.

```python
from P3MLens.Plane import Green
green = Green(box=1000, H=1, p=2, a=6)
lens1 = Plane(coor1, box=1000, m_p=5e2, H=1, p=2, a=6, green=green)
lens2 = Plane(coor2, box=1000, m_p=5e2, H=1, p=2, a=6, green=green)
```
## Author
Kun Xu 

kunxu.sjtu15@foxmail.com
