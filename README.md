# Compressive gate set tomography

This repository contains the Python implementation of a flexible and efficient gate set tomography technique based on low-rank approximations, as described in the following research paper:

**Title**: Compressive gate set tomography

**Authors**: Raphael Brieger, Ingo Roth, and Martin Kliesch

**Link**: https://arxiv.org/abs/2112.05176

## Abstract
Flexible characterization techniques that identify and quantify experimental imperfections under realistic assumptions are crucial for the development of quantum computers. Gate set tomography is a characterization approach that simultaneously and self-consistently extracts a tomographic description of the implementation of an entire set of quantum gates, as well as the initial state and measurement, from experimental data. Obtaining such a detailed picture of the experimental implementation is associated with high requirements on the number of sequences and their design, making gate set tomography a challenging task even for only two qubits.
In this work, we show that low-rank approximations of gate sets can be obtained from significantly fewer gate sequences and that it is sufficient to draw them randomly. Such tomographic information is needed for the crucial task of dealing with coherent noise. To this end, we formulate the data processing problem of gate set tomography as a rank-constrained tensor completion problem. We provide an algorithm to solve this problem while respecting the usual positivity and normalization constraints of quantum mechanics by using second-order geometrical optimization methods on the complex Stiefel manifold. Besides the reduction in sequences, we demonstrate numerically that the algorithm does not rely on structured gate sets or an elaborate circuit design to robustly perform gate set tomography. Therefore, it is more flexible than traditional approaches. We also demonstrate how coherent errors in shadow estimation protocols can be mitigated using estimates from gate set tomography.

## Use Case
This Python package allows users to perform gate set tomography on quantum devices using low-rank approximations, significantly reducing the number of gate sequences required for accurate characterization. It provides an efficient algorithm for solving rank-constrained tensor completion problems and respects the positivity and normalization constraints of quantum mechanics. The algorithm is more flexible than traditional approaches and can be used to mitigate coherent errors in shadow estimation protocols.


## Installation Instructions
To install the package, follow these steps:

### Clone the repository to your local machine:
```bash
git clone git@github.com:rabrie/mGST.git
```

### Change to the repository's directory:
```bash
cd mGST
```

### Install the package using pip:
```bash
pip install .
```

Now you can import the package in your Python scripts and use it. For examples and usage instructions, please refer to the provided Jupyter notebook.