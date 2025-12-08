# LINMA2471 - Optimization Models and Methods II
## Markowitz Portfolio Optimization Project

This project implements and analyzes various optimization methods for solving the Markowitz portfolio optimization problem, as part of the LINMA2471 course at UCLouvain.

---

## 📋 Problem Description

We study two variants of the Markowitz portfolio optimization problem:

### Model 1: Smooth Markowitz (Quadratic)
$$\min_{w \in \Delta_n} \frac{1}{2} w^\top \Sigma w - \lambda \mu^\top w$$

### Model 2: Non-Smooth Markowitz (with Transaction Costs)
$$\min_{w \in \Delta_n} \frac{1}{2} w^\top \Sigma w - \lambda \mu^\top w + c \|w - w_0\|_1$$

Where:
- $w \in \mathbb{R}^n$ — portfolio weights
- $\Delta_n = \{w : w \geq 0, \sum_i w_i = 1\}$ — simplex constraint
- $\Sigma \in \mathbb{R}^{n \times n}$ — covariance matrix of returns
- $\mu \in \mathbb{R}^n$ — expected returns
- $\lambda > 0$ — risk aversion parameter
- $c > 0$ — transaction cost coefficient
- $w_0$ — initial portfolio

---

## Project Structure

```
LINMA2471---Project/
│
├── README.md                 # This file
├── LINMA2471-2025-Homework.pdf  # Project assignement
│
├── python/                   # Main Python package
│   ├── __init__.py
│   ├── main.ipynb           # Main notebook for experiments
│   │
│   ├── models/              # Optimization models
│   │   ├── __init__.py
│   │   └── models.py        # SmoothMarkowitzModel, NonSmoothMarkowitzModel
│   │
│   ├── methods/             # Optimization algorithms
│   │   ├── __init__.py
│   │   └── methods.py       # ProjectedGradient, Subgradient, Proximal, etc.
│   │
│   ├── data/                # Data loading and processing
│   │   ├── __init__.py
│   │   ├── data_processor.py
│   │   └── all_stocks_5yr.csv
│   │
│   └── utils/               # Utility functions
│       ├── __init__.py
│       └── utils.py         # Simplex projection, etc.
│
└── report/                  # Typst report
    ├── main.typ
    └── refs.yml
```

---
## Running

### Prerequisites

- Python 3.10+
- Required packages:

    ```md
    numpy
    pandas
    matplotlib
    ```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DewellGuerand/LINMA2471---Project.git
   cd LINMA2471---Project
   ```

2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib
   ```

   or with conda:

   ```bash
   conda install numpy pandas matplotlib
   ```

### Running the Project

The main experiments are in the Jupyter notebook. From the `python/` directory:

```bash
cd python
jupyter notebook main.ipynb
```

Or open `python/main.ipynb` directly in VS Code.

---

## Implemented Methods

### For Smooth Model (Model 1)

| Method | Description |
|--------|-------------|
| `ProjectedGradientMethod` | Gradient descent with simplex projection |
| `ProjectedGradientDescentMomentum` | Gradient descent with momentum |
| `ProjectedRandomizedCoordinateDescent` | Coordinate descent (O(n) per iteration) |
| `InteriorPointMethod` | Barrier method with Newton steps |

### For Non-Smooth Model (Model 2)

| Method | Description |
|--------|-------------|
| `ProjectedSubgradientMethod` | Subgradient method (constant/diminishing step) |
| `ProximalGradientMethod` | Proximal gradient with soft-thresholding |

---

## Data

The project uses the **S&P 500 stock data** (`all_stocks_5yr.csv`) containing daily prices for ~500 stocks over 5 years (2013-2018).

From this data, we compute:

- **Expected returns** $\mu$: Sample mean of daily returns
- **Covariance matrix** $\Sigma$: Sample covariance of daily returns

---

## Experiments

The notebook contains experiments analyzing:

*TODO*: Add experiments

---

## Authors

- Lucas Ahou
- Guerand Dewell
