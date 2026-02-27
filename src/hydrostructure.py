import numpy as np
import pandas as pd


def generate_structural_data(n_samples=300, random_state=42):
    """
    Generate synthetic fractured-aquifer dataset.
    Creates two structural regimes:
    - 1: fracture corridor
    - 0: matrix domain
    """

    np.random.seed(random_state)

    x = np.random.uniform(0, 100, n_samples)
    y = np.random.uniform(0, 100, n_samples)

    # Define fracture corridor: vertical band
    regime = np.where((x > 40) & (x < 60), 1, 0)

    # Hydraulic transmissivity controlled by regime
    transmissivity = np.where(
        regime == 1,
        np.random.normal(100, 15, n_samples),   # high in fracture
        np.random.normal(20, 5, n_samples)      # low in matrix
    )

    # Synthetic hydrochemical proxy (independent domain)
    hydrochem = np.where(
        regime == 1,
        np.random.normal(5, 1, n_samples),
        np.random.normal(20, 3, n_samples)
    )

    data = pd.DataFrame({
        "x": x,
        "y": y,
        "regime": regime,
        "transmissivity": transmissivity,
        "hydrochem_proxy": hydrochem
    })

    return data
