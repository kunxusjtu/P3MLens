# P3MLens
An Accurate P3M Algorithm for Gravitational Lensing Studies in Simulations.
The code of current version is parallelized using numba based on threads, which may not be suitable for large cosmological simulations. New version based on mpi4py is under development.
\texttt{P3MLens} construct a lens plane from input particles and return the quantities like $\bm{E}$, $\bm{\alpha}$, $\kappa$, $\gamma$ and $\mu$ for positions of interest.
