# Automated Quantum Circuit Design with Nested MCTS

This repository hosts the code and experiment results for our paper [Automated Quantum Circuit Design with Nested Monte Carlo Tree Search](https://arxiv.org/abs/2207.00132) by Peiyong Wang, Muhammad Usman, Udaya Parampalli, Lloyd C. L. Hollenberg and Casey R. Myers.

The environment for this code can be easily installed with docker by running:

`docker pull addwater0315/quantum-research:quantumresearch220405`

Within the `search-scripts` folder, there are Python scripts for running search for the problems demenstrated in our paper. Folder `results_and_plots` contains the results and plots for the paper. `qas` is the main package for our algorithmic framework, in which you can find the implementation of the search algorithm itself as well as code examples for the problems in our paper.

Note:We still need to emphasise that our code is only research-oriented, not for industrial-grade applications.
 It is expected that someone will need basic knowledge of Python programming, including class and inheritance, to modify our code.
