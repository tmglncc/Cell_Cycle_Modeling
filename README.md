# Connecting different approaches for cell cycle modeling: learning ordinary differential equations from individual-based models

We explore the use of cell cycle agent-based models (ABMs) built on the PhysiCell platform alongside their corresponding surrogate ordinary differential equation (ODE) models. We address issues related to the stochastic nature, computational cost, and accuracy of  ABMs in assessing PhysiCell simulations. Considering the potential benefits of surrogate representations as substitutes for costly computational models, we also showcase the use of a sparse identification machine learning tool, namely SINDy-SA, to build surrogate ODE-based representations for ABMs developed on the PhysiCell platform. We show that three key parameters impact both the computational efficiency and accuracy of the PhysiCell simulations: the time step for cell phase transitions, the initial cell count, and the number of replicates. Examining the stochastic characteristics of simulation profiles for cell cycle subpopulation counts and corresponding fractions over time through probability distributions revealed two primary findings: various established probability distributions can appropriately model their dynamics and the convergence to the mean behavior of subpopulation fractions with respect to the number of replicates is faster than that of the subpopulation counts. Finally, the SINDy-SA framework successfully identified the mathematical structures for all employed cell cycle models, even when including apoptosis-induced cell death. We believe that the insights gained from this study can assist researchers in modeling hybrid discrete-continuous multiscale models for tumor growth using the PhysiCell platform and in developing corresponding surrogate representations through the SINDy-SA framework.

![Graphical abstract](https://drive.google.com/uc?export=view&id=1XW2vh6OuLil_fONrHP1b_4zxib3PKBPO)

## Running PhysiCell and SINDy-SA

Our experiments have been performed using the **PhysiCell** platform (version 1.10.4), **SINDy-SA** framework (version 1.0), and Jupyter notebooks. The implementation of the PhysiCell platform is predominantly in C++, while the algorithms for running multiple simulation replicates, gathering statistical data, computing errors and computational costs, assessing the convergence of cell phase solution profiles, and creating various types of plots are developed in Python using Jupyter notebooks. The SINDy-SA framework incorporating the differential evolution algorithm is also built in Python. For instructions on how to use PhysiCell, please visit [https://github.com/MathCancer/PhysiCell](https://github.com/MathCancer/PhysiCell) and look at _QuickStart.md_ and _UserGuide.pdf_ in the _documentation_ folder. For instructions on how to run the SINDy-SA framework and its requirements, please visit [https://github.com/tmglncc/SINDy-SA](https://github.com/tmglncc/SINDy-SA).

## Repository organization

This Github repository is organized according to the following directories:

- **Computational Cost Plot** and **Computational Cost Plot (ic_by_fraction)**: contain a Jupyter notebook to compute the computational cost for multiple experimental settings;
- **Error Plot** and **Error Plot (ic_by_fraction)**: contain Jupyter notebooks to compute the error for multiple experimental settings, based on both cell subpopulation counts and fractions;
- **PhysiCell (template, NULL)** and **PhysiCell (template, NULL, ic_by_fraction)**: contain a PhysiCell project modeling cell cycle dynamics, including the mechanism of apoptosis-induced cell death, and Jupyter notebooks for running multiple simulation replicates, gathering statistical data, assessing the convergence of cell phase solution profiles, and creating various types of plots.
- **SINDy-SA** folders: apply the SINDy-SA framework to build a surrogate ordinary differential equation models from PhysiCell simulation data for multiple scenarios.

Of note, **ic_by_fraction** denotes experiments in which the ABM initial condition is defined based on percentages of cell phase fractions.

## Cite as

Naozuka, G.T.; Rocha, H.L.; Pereira, T.J.; Libotte, G.B.; Almeida, R.C. Cell Cycle Modeling, 2024. Version 1.0. Available online: [https://github.com/tmglncc/Cell_Cycle_Modeling](https://github.com/tmglncc/Cell_Cycle_Modeling) (accessed on 25 January 2024), doi: [10.5281/zenodo.10569158](https://zenodo.org/doi/10.5281/zenodo.10569158).
