## Ultrafast critical ground state preparation via bang-bang protocols

Code, notebooks, and data used to produce the results presented in:

> Luca Innocenti, Gabriele De Chiara, Mauro Paternostro, Ricardo Puebla (2020). "*Ultrafast critical ground state preparation via bang-bang protocols*" ([arXiv:2007.07381](https://arxiv.org/abs/2007.07381)).

# Content

All the data is in `./results`. The majority of it is designed to be read via the `ipynb` notebooks here.
The notebooks require code found in `./src`. Running the code doesn't require to install the module `./src` system-wide, but the notebooks do expect the directory structure to be preserved. Running the notebooks therefore requires downloading the full repository and preserving its directory structure.

## Guide to the notebooks

- `./lz_optimizations.ipynb`
    Optimizations to reach the ground state at the avoided crossing of the LZ.
    Contains optimizations with double-bang and CRAB protocols, as well as the code used to generate the corresponding figures in the paper.
    Presented attempts at both generating ground state at the critical point, and generating ground state in the other phase.
- `./lmg_4spins_optimizations.ipynb`, `./lmg_20spins_optimizations.ipynb`, `./lmg_50spins_optimizations.ipynb`
    Optimizations double-bang vs CRAB for LMG with 20 and 50 spins.
    The 50 spins notebook also contains optimisations for different energy bounds.
- `./rabi_and_lmg_optimizations.ipynb`
    Older notebook. Contains a variety of optimisation attempts with LMG and Rabi models.
