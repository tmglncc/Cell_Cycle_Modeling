# Connecting different approaches for cell cycle modeling: learning ordinary differential equations from individual-based models

*Background and Objectives*: We explore the use of cell cycle agent-based models (ABMs) built on the PhysiCell platform alongside their corresponding surrogate ordinary differential equation (ODE) models. We also address issues related to the stochastic nature, computational cost, and accuracy of  ABMs in assessing PhysiCell simulations.

*Methods*: Considering the potential benefits of surrogate representations as substitutes for costly computational models, we showcase the use of a sparse identification machine learning tool, namely SINDy-SA, to build surrogate ODE-based representations for ABMs developed on the PhysiCell platform.

*Results*: We show that three key parameters impact both the computational efficiency and accuracy of the PhysiCell simulations: the time step for cell phase transitions, the initial cell count, and the number of replicates. Examining the stochastic characteristics of simulation profiles for cell cycle subpopulation counts and corresponding fractions over time through probability distributions revealed two primary findings: various established probability distributions can appropriately model their dynamics and the convergence to the mean behavior of subpopulation fractions with respect to the number of replicates is faster than that of the subpopulation counts. Finally, the SINDy-SA framework successfully identified the mathematical structures for all employed cell cycle models, even when including apoptosis-induced cell death.

*Conclusions*: We believe that the insights gained from this study can assist researchers in modeling hybrid discrete-continuous multiscale models for tumor growth using the PhysiCell platform and in developing corresponding surrogate representations through the SINDy-SA framework.

![Graphical abstract](https://drive.google.com/uc?export=view&id=1XW2vh6OuLil_fONrHP1b_4zxib3PKBPO)

## Requirements and running PhysiCell and SINDy-SA

Our experiments have been performed using the **PhysiCell** platform (version 1.10.4), **SINDy-SA** framework (version 1.0), and Jupyter notebooks (package version 4.1.4). The implementation of the PhysiCell platform is predominantly in C++, while the algorithms for running multiple simulation replicates, gathering statistical data, computing errors and computational costs, assessing the convergence of cell phase solution profiles, and creating various types of plots are developed in Python using Jupyter notebooks. The SINDy-SA framework incorporating the differential evolution algorithm is also built in Python.

For requirements and instructions on how to use PhysiCell, please access [https://github.com/MathCancer/PhysiCell](https://github.com/MathCancer/PhysiCell) and look at _QuickStart.md_ and _UserGuide.pdf_ in the _documentation_ folder.

For requirements and instructions on how to run the SINDy-SA framework, please access [https://github.com/tmglncc/SINDy-SA](https://github.com/tmglncc/SINDy-SA).

## Repository organization

This GitHub repository is organized according to the following directories and files:

- **Computational Cost Plot** and **Computational Cost Plot (ic_by_fraction)** folders: contain _plot_cost.ipynb_ to plot the computational costs for multiple experimental settings.
- **Error Plot** and **Error Plot (ic_by_fraction)** folders: contain _plot_error_by_fraction.ipynb_ and _plot_error_by_population.ipynb_ to plot the sum of squared errors (SSE) up to the final simulation time for tumor cell fractions/populations between the standard ODE-based model solution and PhysiCell simulations averaged across replicates.
- **PhysiCell (template, NULL)** and **PhysiCell (template, NULL, ic_by_fraction)** folders: contain a PhysiCell project modeling the cell cycle dynamics, including the mechanism of apoptosis-induced cell death. They also contain the following Python scripts and Jupyter notebooks:
  - _run_replicates.py_: runs multiple simulation replicates;
  - _plot_stochastic_by_population.ipynb_: gathers statistical information (mean and standard deviation) regarding the dynamics of tumor cell subpopulations;
  - _plot_stochastic_by_fraction.ipynb_: gathers statistical information (mean and standard deviation) regarding the dynamics of tumor cell fractions;
  - _compute_cost.ipynb_: computes the computational costs;
  - _compute_error.ipynb_: computes the sum of squared errors (SSE) up to the final simulation time for tumor cell fractions between the standard ODE-based model solution and PhysiCell simulations averaged across replicates;
  - _plot_distribution.ipynb_: constructs frequency histograms, calibrates distribution parameters, and generates essential metrics such as the sum of squared errors (SSE) and model selection criteria. These criteria include first-order Akaike (AIC), second-order Akaike (AICc), and Bayesian (BIC) information criteria. It also assesses the weights of the selection criteria that indicate the best probability distribution based on the data;
  - _plot_kl_divergence.ipynb_: compares kernel density approximations of solution profiles at the final simulation time of each experimental setting with the corresponding reference experiments featuring R = 300 replicates;
  - _plot_heatmap.ipynb_: plots heatmaps of Kullback-Leibler divergence.
- **SINDy-SA** folders: apply the SINDy-SA framework to build surrogate ODE-based models from PhysiCell simulation data for multiple cell cycle scenarios and experimental settings.

Of note, **ic_by_fraction** indicates experiments in which the ABM initial condition is defined based on percentages of cell phase fractions.

To facilitate the visualization and extraction of PhysiCell simulation data, we also include the employed versions of **PhysiCell-Studio** and **Python-loader** in this GitHub repository.

## Cite as

Naozuka, G.T.; Rocha, H.L.; Pereira, T.J.; Libotte, G.B.; Almeida, R.C. Cell Cycle Modeling, 2024. Version 1.0. Available online: [https://github.com/tmglncc/Cell_Cycle_Modeling](https://github.com/tmglncc/Cell_Cycle_Modeling) (accessed on 25 January 2024), doi: [10.5281/zenodo.10569158](https://zenodo.org/doi/10.5281/zenodo.10569158).
